import json
import re
from pathlib import Path
from typing import Any, TypedDict

from browser_env import Action, ActionParsingError, Trajectory
from browser_env.env_config import URL_MAPPINGS
from browser_env.utils import StateInfo
from llms import lm_config
from llms.tokenizers import Tokenizer
from llms.utils import APIInput
from rank_bm25 import BM25Okapi
import random
import os
import networkx as nx
import csv


def pick_ranking_random_action(action_dict: dict[str, int], chosen_count: int) -> str:
    """
    从 action_dict 中按照 count 加权随机取 chosen_count 个 action
    如果 action_dict 小于等于 chosen_count 个，则全要
    如果 action_dict 大于 chosen_count 个，则按照 count 加权随机取 chosen_count 个，但是每个 action 只能被取一次
    返回一个 list，list 中是 action 的 list
    """
    if len(action_dict) <= chosen_count:
        return list(action_dict.keys())
    else:
        action_list = []
        for _ in range(chosen_count):
            action = random.choices(list(action_dict.keys()), weights=list(action_dict.values()))[0]
            action_list.append(action)
            action_dict[action] = 0
            action_dict = {k: v for k, v in action_dict.items() if v > 0}
        return action_list


class Instruction(TypedDict):
    """Instruction for constructing prompt"""

    intro: str
    examples: list[tuple[str, str]]
    template: str
    meta_data: dict[str, Any]


class PromptConstructor(object):
    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        self.instruction_path = Path(instruction_path)
        self.obs_modality = "text"
        self.lm_config = lm_config
        instruction = json.load(open(self.instruction_path))
        instruction["examples"] = [tuple(e) for e in instruction["examples"]]
        self.instruction: Instruction = instruction
        self.tokenizer = tokenizer

    def get_lm_api_input(
        self, intro: str, examples: list[tuple[str, str]], current: str
    ) -> APIInput:

        """Return the require format for an API"""
        message: list[dict[str, str]] | str
        if "openai" in self.lm_config.provider:
            if self.lm_config.mode == "chat":
                message = [{"role": "system", "content": intro}]
                for (x, y) in examples:
                    message.append(
                        {
                            "role": "system",
                            "name": "example_user",
                            "content": x,
                        }
                    )
                    message.append(
                        {
                            "role": "system",
                            "name": "example_assistant",
                            "content": y,
                        }
                    )
                message.append({"role": "user", "content": current})
                return message
            elif self.lm_config.mode == "completion":
                message = f"{intro}\n\n"
                message += "Here are a few examples:\n"
                for example in examples:
                    message += f"Observation\n:{example[0]}\n\n"
                    message += f"Action: {example[1]}\n\n"
                message += "Now make prediction given the observation\n\n"
                message += f"Observation\n:{current}\n\n"
                message += "Action:"
                return message
            else:
                raise ValueError(
                    f"OpenAI models do not support mode {self.lm_config.mode}"
                )
        elif "huggingface" in self.lm_config.provider:
            # https://huggingface.co/blog/llama2#how-to-prompt-llama-2
            # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L320
            if "Llama-2" in self.lm_config.model:
                if self.lm_config.mode == "chat":
                    B_INST, E_INST = "[INST]", "[/INST]"
                    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                    BOS, EOS = "<s>", "</s>"
                    # adding the system message to be the starting of the first example
                    examples = [
                        (
                            B_SYS + intro + E_SYS + examples[0][0],
                            examples[0][1],
                        )
                    ] + examples[1:]
                    message = "".join(
                        [
                            f"{BOS}{B_INST} {x.strip()} {E_INST} {y.strip()} {EOS}"
                            for (x, y) in examples
                        ]
                    )
                    # add the current observation
                    message += f"{BOS}{B_INST} {current.strip()} {E_INST} {self.instruction['meta_data'].get('force_prefix', '')}"

                    return message
                else:
                    raise ValueError("Only chat mode is supported for Llama-2")
            else:
                raise ValueError(
                    f"Huggingface models do not support model_tag {self.lm_config.gen_config['model_tag']}"
                )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        raise NotImplementedError

    def map_url_to_real(self, url: str) -> str:
        """Map the urls to their real world counterparts"""
        for i, j in URL_MAPPINGS.items():
            if i in url:
                url = url.replace(i, j)
        return url

    def map_url_to_local(self, url: str) -> str:
        """Map the urls to their local counterparts"""
        for i, j in URL_MAPPINGS.items():
            if j in url:
                url = url.replace(j, i)
            # https
            if j.replace("http", "https") in url:
                url = url.replace(j.replace("http", "https"), i)
        return url

    def _extract_action(self, response: str) -> tuple[str, str]:
        raise NotImplementedError

    def extract_action(self, response: str) -> tuple[str, str]:
        response, intention = self._extract_action(response)
        response = self.map_url_to_local(response)
        return response, intention


class DirectPromptConstructor(PromptConstructor):
    """The agent will direct predict the action"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        """Construct prompt given the trajectory"""
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]

        # input x
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        # make sure all keywords are replaced
        assert all([f"{{k}}" not in current for k in keywords])
        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> tuple[str, str]:
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip(), ""
        else:
            raise ActionParsingError(
                f"Cannot parse action from response {response}"
            )


print("start to load triplets")
with open("/home/zjusst/qms/webarena/result_stage_1_explore_v2/flitered_triplets.csv", "r") as f:
    reader = csv.reader(f)
    # 跳过第一行
    next(reader)
    triplets = [row for row in reader]

intention_corpus = [t[2] for t in triplets]

page_positions = set()
for triplet in triplets:
    page_positions.add(triplet[0])
    page_positions.add(triplet[3])

# 转为 list
page_positions = list(page_positions)
page_corpus = []

for page_position in page_positions:
    file_name, traj_index, pos, real_url = eval(page_position)
    with open(f"/home/zjusst/qms/webarena/result_stage_1_explore_v2/add_local_intention_trajs/{file_name}", "r") as f:
        trajs = json.load(f)
    traj = trajs[traj_index]
    if pos == 0:
        a11y = traj["a11y_before"]
    else:
        a11y = traj["a11y_after"]
    page_corpus.append(a11y)

G = nx.MultiDiGraph()
for source, action, intent, target in triplets:
    # 将动作和意图作为边的属性存储
    G.add_edge(source, target, action=action, intent=intent)

print("complete to load corpus and graph")



def bm25_retrieval(query, corpus, top_n=3):
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    top_n_docs = bm25.get_top_n(tokenized_query, corpus, n=top_n)
    top_n_docs_index = [corpus.index(doc) for doc in top_n_docs]
    return top_n_docs_index


def get_guidelines(a11y_tree: str, plan: str, save_path: str) -> str:
    page_indexs = bm25_retrieval(a11y_tree, page_corpus, top_n=3)
    retrived_page_positions = [str(page_positions[i]) for i in page_indexs]
    intention_indexs = bm25_retrieval(plan, intention_corpus, top_n=3)
    retrived_intention_corpus = [triplets[i][2] for i in intention_indexs]

    if len(retrived_page_positions) == 0 or len(retrived_intention_corpus) == 0:
        return ""

    guidelines = []

    for page_position in retrived_page_positions:
        for target_intent in retrived_intention_corpus:

            if len(guidelines) >= 3:
                break

            candidate_edges = []
            for u, v, data in G.edges(data=True):
                if data.get('intent') == target_intent:
                    candidate_edges.append({'source': u, 'target': v, 'data': data})
                    break
            
            if not candidate_edges:
                continue
            edge = candidate_edges[0]
            path_destination_node = edge['source']

            try:
                node_path = nx.shortest_path(G, source=page_position, target=path_destination_node)
                trajectory = []
                # 遍历路径中的每一步（除最后一步外）
                for i in range(len(node_path) - 1):
                    u, v = node_path[i], node_path[i+1]
                    # 获取这两个节点间边的信息（这里我们取第一条，因为是最短路径）
                    edge_data = G.get_edge_data(u, v)[0] 
                    trajectory.append((u, edge_data['action'], edge_data['intent']))
            
                # 添加最后的目标边，完成轨迹
                final_step = (edge['source'], edge['data']['action'], edge['data']['intent'])
                trajectory.append(final_step)
            
                # 添加最终到达的页面
                trajectory.append((edge['target'], None, None)) 
                guidelines.append(trajectory)
            except nx.NetworkXNoPath:
                continue
            except nx.NodeNotFound:
                continue

    if len(guidelines) == 0:
        return ""

    ret = ""
    for guideline in guidelines:
        for step in guideline:
            page, action, intent = step
            _, _, _, real_url = eval(page)
            if action:
                ret += f"{real_url} with action {action} and intention {intent} --> "
            else:
                ret += f"{real_url}"
        ret += "\n"
    ret += "\n"
    return ret


class CoTPromptConstructor(PromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]


    def get_this_session_history_actions(self, url: str, save_path: str) -> str:

        if not os.path.exists(os.path.join(save_path, "history_url_and_action.csv")):
            return ""
        
        this_time_url_and_actions = []
        with open(os.path.join(save_path, "history_url_and_action.csv"), "r") as f:
            for line in f:
                if "stop [Early stop:" in line:
                    this_time_url_and_actions.clear()
                else:
                    url, real_url, action = line.split("#####")
                    this_time_url_and_actions.append((real_url, action))
        ret = ""
        for real_url, action in this_time_url_and_actions:
            ret += f"{real_url};{action}"
        return ret


    def get_history_actions(self, url: str, save_path: str, this_time: bool) -> str:
        """
        在 save_path/history_url_and_action.csv 中
        格式是 url#####real_url#####action
        找到所有 url 为 url 的行，格式化为字符串，格式为
        ```
        click [id] where [id] is link 'Forums';
        click [id] where [id] is link 'Wiki';
        ```
        """

        if this_time:
            return self.get_this_session_history_actions(url, save_path)

        all_real_urls = {}  # key: real_url, value: cnt
        all_actions = {}  # key: "action", value: "count"
        with open(os.path.join(save_path, "history_url_and_action.csv"), "r") as f:
            for line in f:
                try:
                    old_url, real_url, action = line.split("#####")
                    all_real_urls[real_url] = all_real_urls.get(real_url, 0) + 1
                    
                    # 对 action 需要特殊处理，替换所有 [数字] 为 [id]
                    action = re.sub(r'\[(\d+)\]', r'[id]', action)
                    all_actions[action] = all_actions.get(action, 0) + 1

                except Exception as e:
                    continue
        # 按照 cnt 为权重随机选20个
        visited_real_urls = pick_ranking_random_action(all_real_urls, 20)
        visited_actions = pick_ranking_random_action(all_actions, 20)
        
        ret_list = {}  # key: "action", value: "count"
        with open(os.path.join(save_path, "history_url_and_action.csv"), "r") as f:
            for line in f:
                try:
                    old_url, real_url, action = line.split("#####")
                    if old_url == url:
                        if action not in ret_list:
                            ret_list[action] = 0
                        ret_list[action] += 1
                except Exception as e:
                    continue

        this_time_actions = []
        with open(os.path.join(save_path, "history_url_and_action.csv"), "r") as f:
            # 找到最底下的 stop [Early stop: Reach max steps一行，从这里再往下读
            for line in f:
                if "stop [Early stop:" in line:
                    this_time_actions.clear()
                else:
                    this_time_actions.append(line.split("#####")[2])

        ret_list = pick_ranking_random_action(ret_list, 20)
        
        ret = ""
        for real_url in visited_real_urls:
            ret += f"{real_url};\n"
        ret += "\n"
        for action in visited_actions:
            ret += f"{action};"
        ret += "\n"
        for page in this_time_actions:
            ret += f"{page}"
        ret += ";\n"
        for action in ret_list:
            ret += f"{action}"
        return ret
            

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        save_path = meta_data["save_path"]

        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]


        if "plan" in meta_data and meta_data["plan"] != "":
            plan = meta_data["plan"]
            current = template.format(
                objective=intent,
                url=self.map_url_to_real(url),
                observation=obs,
                plan=plan,
                guidelines=get_guidelines(obs, plan, save_path),
                history=self.get_history_actions(url, save_path, this_time=True),
                previous_action=previous_action_str,
            )
        else:
            current = template.format(
                objective=intent,
                url=self.map_url_to_real(url),
                observation=obs,
                history=self.get_history_actions(url, save_path, this_time=False),
                previous_action=previous_action_str,
            )

        
        with open(os.path.join(save_path, "history_url_and_action.csv"), "a") as f:
            f.write(f"{url}#####{self.map_url_to_real(url)}#####")

        assert all([f"{{k}}" not in current for k in keywords])

        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> tuple[str, str]:
        # find the first occurence of action
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"  # ((.|\n)*?) 代表任意字符和换行符，*? 代表非贪婪匹配
        match = re.search(pattern, response)
        if match:
            json_str = match.group(1).strip()
            try:
                json_obj = json.loads(json_str)
                return json_obj["action"], json_obj["intention"]
            except Exception as e:
                raise ActionParsingError(
                    f"Cannot parse action from response {response}"
                )
        else:
            raise ActionParsingError(
                f'Cannot find the answer phrase "{self.answer_phrase}" in "{response}"'
            )

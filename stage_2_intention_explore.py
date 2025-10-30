"""Script to run end-to-end evaluation on the benchmark"""
import argparse
import glob
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
import csv
import openai

from agent import (
    Agent,
    PromptAgent,
    TeacherForcingAgent,
    construct_agent,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router
from browser_env.env_config import URL_MAPPINGS
import copy

from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)

os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGSMITH_ENDPOINT'] = ''
os.environ['LANGSMITH_API_KEY'] = ''
os.environ['LANGSMITH_PROJECT'] = ''


LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


#====================================================


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When concesecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When concesecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )
    parser.add_argument(
        "--model_endpoint",
        help="huggingface model endpoint",
        type=str,
        default="",
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=1000)

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type != "accessibility_tree"
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to early stop"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    # Case: go to other websites
    last_state = trajectory[-1]
    if "10.130.138.30" not in last_state["info"]["page"].url:
        return True, f"Go to other websites, current url: {last_state['info']['page'].url}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""


def map_url_to_real(url: str) -> str:
    for i, j in URL_MAPPINGS.items():
        if i in url:
            url = url.replace(i, j)
    return url

def map_url_to_local(url: str) -> str:
    for i, j in URL_MAPPINGS.items():
        if j in url:
            url = url.replace(j, i)
        if j.replace("http", "https") in url:
            url = url.replace(j.replace("http", "https"), i)
    return url


def save_trajectory(trajectory, save_path, intent):

    # 如果 path 所在文件夹不存在，则创建
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    traj_save = []
    for i, item in enumerate(trajectory):
        # 如果是状态，即 item[0] 为 dict 且包含 'observation' 和 'info' 键
        if isinstance(item[0], dict) and 'observation' in item[0] and 'info' in item[0]:
            continue
        # 如果是动作，即 item[0] 为 dict 且包含 'raw_prediction' 且 i+1
        elif isinstance(item[0], dict) and 'raw_prediction' in item[0] and i+1 < len(trajectory):

            temp_action = item[0]
            temp_action['coords'] = temp_action['coords'].tolist()

            step_data = {
                "url_before": trajectory[i-1][0]['info']["page"].url,
                "url_after": trajectory[i+1][0]['info']["page"].url,
                "url_real_before": trajectory[i-1][1],
                "url_real_after": trajectory[i+1][1],
                "a11y_before": trajectory[i-1][0]['observation']['text'],
                "a11y_after": trajectory[i+1][0]['observation']['text'],
                "state_before": trajectory[i-1][0]['info']['observation_metadata'],
                "state_after": trajectory[i+1][0]['info']['observation_metadata'],
                "action": temp_action,
                "reasoning": temp_action['raw_prediction'],
                "action_str": item[1],
                "forward_intention": item[2],
                "plan": item[3],
                "intent": intent
            }
            traj_save.append(step_data)

    with open(save_path, "w") as f:
        json.dump(traj_save, f, indent=4)

def get_planer_history(save_path):
        this_time_real_url_and_actions = []
        if not os.path.exists(os.path.join(save_path, "history_url_and_action.csv")):
            return ""
        with open(os.path.join(save_path, "history_url_and_action.csv"), "r") as f:
            # 找到最底下的 stop [Early stop: Reach max steps一行，从这里再往下读
            for line in f:
                if "stop [Early stop:" in line:
                    this_time_real_url_and_actions.clear()
                else:
                    url, real_url, action = line.split("#####")
                    this_time_real_url_and_actions.append((real_url, action))

        ret = ""
        for real_url, action in this_time_real_url_and_actions:
            ret += f"{real_url};{action}"
        return ret


def generate_plan(trajectory, intent, meta_data):
    system_prompt = """You are an Expert Web Strategist, acting as the high-level reasoning module for a web agent. Your goal is to break down a complex user objective into sub-task or "plan" that should be accomplished next. This plan is NOT a specific action, but a conceptual sub-goal that a separate executor agent will then carry out.

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current web page's accessibility tree: A simplified representation of the webpage.
The current web page's URL: The page you're currently on.
The open tabs: A list of your open tabs.
A list of visited URLs and Actions: A list of URLs and Actions you have already explored in this session.

Note:
1. Since you can't see future webpages, each sub-goal should be abstract, high-level, and not involve interacting with specific UI elements.
2. You should use the current page as the starting point to explore the website. You can first analyze the provided accessibility tree to understand what is possible and relevant from the user's current position before formulating your plan.
3. Use the list of visited URLs and past actions to inform your strategy. Avoid formulating a plan that would lead to repeating a failed action or getting stuck in a navigation loop.
4. If the user's objective is completed, you should suggest to end the session. And you should suggest next sub-goals when one same action is repeated multiple times (eg. use search too many times).

Your response should be 2~3 simple sub-goals that describe what to do next to achieve the user's objective."""

    user_example_prompt = """OBSERVATION:
[164] textbox 'Search' focused: True required: False
[171] button 'Go'
[174] link 'Find directions between two points'
[212] heading 'Search Results'
[216] button 'Close'
URL: http://openstreetmap.org
OBJECTIVE: Show me the restaurants near CMU
HISTORY: click [123] where [123] is link 'openstreetmap.org'
PREVIOUS ACTION: None"""

    ai_example_prompt = """1. Use the search functionality to find "restaurants near CMU".
2. Analyze the search results to locate relevant restaurant information on the map.
3. Examine the details of a specific restaurant to confirm its location and type."""

    user_prompt_template = """OBSERVATION:
{observation}
URL: {url}
OBJECTIVE: {objective}
HISTORY: {history}
PREVIOUS ACTION: {previous_action}"""

    state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
    obs = state_info["observation"]["text"]
    page = state_info["info"]["page"]
    previous_action_str = meta_data['action_history'][-1]
    save_path = meta_data['save_path']

    user_prompt = user_prompt_template.format(
        observation=obs,
        url=map_url_to_real(page.url),
        objective=intent,
        history=get_planer_history(save_path),
        previous_action=previous_action_str
    )

    message = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "name": "example_user", "content": user_example_prompt},
        {"role": "system", "name": "example_assistant", "content": ai_example_prompt},
        {"role": "user", "content": user_prompt},
    ]

    with open(os.path.join(save_path, "prompt_and_response.log"), "a") as f:
        for m in message:
            f.write(str(m['content']) + "\n\n")

    llm_config = lm_config.construct_llm_config(args)
    n = 0


    response = call_llm(llm_config, message)
    with open(os.path.join(save_path, "prompt_and_response.log"), "a") as f:
        f.write(str(response) + "\n\n\n")
   
    return response


#====================================================


args = config()
args.sleep_after_execution = 2.0
prepare(args)

test_file_list = []
st_idx = args.test_start_idx
ed_idx = args.test_end_idx
for i in range(st_idx, ed_idx):
    test_file_list.append(f"config_files/{i}.json")
# if "debug" not in args.result_dir:
#     test_file_list = get_unfinished(test_file_list, args.result_dir)

if len(test_file_list) == 0:
    logger.info("No task left to run")
    exit(0)


print(f"Total {len(test_file_list)} tasks left")
args.render = False
args.render_screenshot = True
args.save_trace_enabled = True

args.current_viewport_only = True
dump_config(args)

agent = construct_agent(args)


scores = []
max_steps = args.max_steps

early_stop_thresholds = {
    "parsing_failure": args.parsing_failure_th,
    "repeating_action": args.repeating_action_failure_th,
}


env = ScriptBrowserEnv(
    headless=not args.render,
    slow_mo=args.slow_mo,
    observation_type=args.observation_type,
    current_viewport_only=args.current_viewport_only,
    viewport_size={
        "width": args.viewport_width,
        "height": args.viewport_height,
    },
    save_trace_enabled=args.save_trace_enabled,
    sleep_after_execution=args.sleep_after_execution,
)

config_file_list = test_file_list

#======================================================================

for config_file in config_file_list:

    render_helper = RenderHelper(
        config_file, args.result_dir, args.action_set_tag
    )

    # get intent
    with open(config_file) as f:
        _c = json.load(f)
        intent = _c["intent"]
        task_id = _c["task_id"]
        # automatically login
        if _c["storage_state"]:
            cookie_file_name = os.path.basename(_c["storage_state"])
            comb = get_site_comb_from_filepath(cookie_file_name)
            temp_dir = tempfile.mkdtemp()
            # subprocess to renew the cookie
            subprocess.run(
                [
                    "python",
                    "browser_env/auto_login.py",
                    "--auth_folder",
                    temp_dir,
                    "--site_list",
                    *comb,
                ]
            )
            _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
            assert os.path.exists(_c["storage_state"])
            # update the config file
            config_file = f"{temp_dir}/{os.path.basename(config_file)}"
            with open(config_file, "w") as f:
                json.dump(_c, f)

    logger.info(f"[Config file]: {config_file}")
    logger.info(f"[Intent]: {intent}")

    agent.reset(config_file)
    trajectory: Trajectory = []
    to_save_trajectory = []
    obs, info = env.reset(options={"config_file": config_file})
    state_info: StateInfo = {"observation": obs, "info": info}
    trajectory.append(state_info)
    to_save_trajectory.append((copy.deepcopy(state_info), map_url_to_real(state_info["info"]["page"].url)))

    meta_data = {
        "action_history": ["None"],
        "save_path": args.result_dir,
        "plan": ""
    }

    #====================================================================

    with open(os.path.join(args.result_dir, "prompt_and_response.log"), "a") as f:
        f.write("==================================================================== \n")
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
        f.write("start recording \n")

    while True:

        early_stop_flag, stop_info = early_stop(
            trajectory, max_steps, early_stop_thresholds
        )

        if early_stop_flag:
            action = create_stop_action(f"Early stop: {stop_info}")
            forward_intention = "None"
        else:
            try:
                # plan prompt
                plan = generate_plan(trajectory, intent, meta_data=meta_data)
                meta_data['plan'] = plan

                action, forward_intention = agent.next_action(
                    trajectory, intent, meta_data=meta_data
                )
            except ValueError as e:
                # get the error message
                action = create_stop_action(f"ERROR: {str(e)}")
                forward_intention = "None"

        trajectory.append(action)

        #########################################################################

        action_str = get_action_description(
            action,
            state_info["info"]["observation_metadata"],
            action_set_tag=args.action_set_tag,
            prompt_constructor=agent.prompt_constructor
            if isinstance(agent, PromptAgent)
            else None,
        )
        render_helper.render(
            action, state_info, meta_data, args.render_screenshot
        )
        meta_data["action_history"].append(action_str)

        with open(os.path.join(args.result_dir, "history_url_and_action.csv"), "a") as f:
            # 如果包含换行符
            if "\n" in action_str or "The previous prediction you issued was" in action_str:
                f.write(f"invalid action_str\n")
            else:
                f.write(f"{action_str}\n")

        to_save_trajectory.append((action, action_str, forward_intention, plan))

        if action["action_type"] == ActionTypes.STOP:
            break

        obs, _, terminated, _, info = env.step(action)
        # _, real_info = get_state(env)
        state_info = {"observation": obs, "info": info}
        trajectory.append(state_info)
        to_save_trajectory.append((copy.deepcopy(state_info), map_url_to_real(state_info["info"]["page"].url)))

        if terminated:
            # add a action place holder
            trajectory.append(create_stop_action(""))
            to_save_trajectory.append(create_stop_action(""))
            break


    index = 0
    trace_file_path = Path(args.result_dir) / "traces" / f"{task_id}.zip"
    if trace_file_path.exists():
        index = 1
        while True:
            new_trace_file_path = Path(args.result_dir) / "traces" / f"{task_id}_{index}.zip"
            if new_trace_file_path.exists():
                index += 1
            else:
                trace_file_path = new_trace_file_path
                break

    env.save_trace(trace_file_path)

    if index == 0:
        save_trajectory(to_save_trajectory, Path(args.result_dir) / "trajs" / f"{task_id}.json", intent)
    else:
        save_trajectory(to_save_trajectory, Path(args.result_dir) / "trajs" / f"{task_id}_{index}.json", intent)

    # import pickle
    # trajectory_save_path = Path(args.result_dir) / "trajs" / f"{task_id}_{index}.pkl"
    # with trajectory_save_path.open("wb") as f:
    #     pickle.dump(to_save_trajectory, f)

    render_helper.close()

#####################################################
env.close()
import networkx as nx
from pyvis.network import Network

# --------------------------------------------------------------------------
# 1. 你的三元组数据 (用示例数据代替)
# 在实际使用中，你会从文件中加载这些数据
# 格式: (前页面标识, 动作描述, 后页面标识)
# 页面标识可以是页面的URL、截图文件的哈希值或你定义的任何唯一ID
# triplets = [
#     ('HomePage', 'click("Login Button")', 'LoginPage'),
#     ('HomePage', 'click("Products Link")', 'ProductsPage'),
#     ('ProductsPage', 'click("Item 1")', 'Item1_DetailsPage'),
#     ('ProductsPage', 'click("Item 2")', 'Item2_DetailsPage'),
#     ('Item1_DetailsPage', 'click("Add to Cart")', 'ShoppingCartPage'),
#     ('Item2_DetailsPage', 'click("Add to Cart")', 'ShoppingCartPage'),
#     ('LoginPage', 'type("username & password")', 'HomePage'), # 登录后回到主页
#     ('ShoppingCartPage', 'go_back()', 'ProductsPage') # 从购物车返回
# ]
# --------------------------------------------------------------------------



with open('history_pages_and_action.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

triplets = []
last_page = None
last_action = None

for line in lines:
    if line.strip():
        if "stop [Early stop: Reach max steps 10]" in line:
            last_page = None
            last_action = None
            continue

        # 分隔符为 #####
        parts = line.split('#####')
        if len(parts) == 3:
            this_page = parts[1].strip()
            this_action = parts[2].strip()
            if last_page:
                triplets.append((last_page, last_action, this_page))
            last_page = this_page
            last_action = this_action

# 2. 使用 NetworkX 创建一个有向图 (因为 A->B 和 B->A 是不同的)
G = nx.DiGraph()

# 3. 遍历三元组数据，填充图的节点和边
for source, action, target in triplets:
    # 添加节点 (如果节点已存在，NetworkX会自动忽略)
    # 我们可以为节点添加属性，例如'title'，当鼠标悬停时会显示
    G.add_node(source, title=f"Page ID: {source}")
    G.add_node(target, title=f"Page ID: {target}")

    # 添加边，并把动作描述作为边的标签
    G.add_edge(source, target, label=action)


# --------------------------------------------------------------------------
# 4. 使用 Pyvis 进行交互式可视化
# 创建一个Pyvis网络图对象
# notebook=True 如果你在Jupyter Notebook中使用
# directed=True 会让边带上箭头，非常重要
net = Network(height='2000px', width='100%', notebook=True, directed=True)

# 从 NetworkX 对象加载图数据
net.from_nx(G)

# 为可视化添加一些物理引擎的选项，让布局更好看
# net.toggle_physics(True)

# 手动设定物理引擎参数
net.set_options("""
{
  "physics": {
    "enabled": true,
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {
      "springLength": 600
    }
  }
}
""")

# 添加交互UI的选项，例如筛选节点
#net.show_buttons(filter_=['physics'])

# 5. 生成HTML文件
# 这个文件将保存在你的脚本所在的目录下
# 你可以直接在浏览器中打开它
net.show('interactive_graph.html')

print("成功生成 'interactive_graph.html' 文件！请在浏览器中打开查看。")
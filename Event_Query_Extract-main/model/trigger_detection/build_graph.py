import json

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain

def print_graph_detail(graph):
    """
    格式化显示Graph参数
    :param graph:
    :return:
    """
    dst = {"nodes"    : nx.number_of_nodes(graph),
           "edges"    : nx.number_of_edges(graph),
           "selfloops": nx.number_of_selfloops(graph),
           "isolates" : nx.number_of_isolates(graph),
           "覆盖度"      : 1 - nx.number_of_isolates(graph) / nx.number_of_nodes(graph), }
    print_table(dst)

def print_table(dst):
    table_title = list(dst.keys())
    from prettytable import PrettyTable
    table = PrettyTable(field_names=table_title, header_style="title", header=True, border=True,
                        hrules=1, padding_width=2, align="c")
    table.float_format = "0.4"
    table.add_row([dst[i] for i in table_title])
    print(table)

# G = nx.Graph()
# file_path = './data/DuEE1.0/_trainevent_type_cross.txt'
# edgelist = []
# node_set = set()
# total_event = []
# event_id = {}
# scheme_path = './data/DuEE1.0/event_schema.json'
# with open(scheme_path, 'r', encoding='utf-8') as j:
#     for line in j:
#         context = json.loads(line)
#         total_event.append(context['event_type'])
# for index, event_type in enumerate(total_event):
#     event_id[event_type] = index
# with open(file_path, 'r', encoding='utf-8') as f:
#     next(f)
#     for context in f:
#         if context != '}':
#             context = context.replace("'('",'')
#             context = context.replace("'", '')
#             context = context.replace(")", '')
#             key, index = context.split(':')
#             index = index.replace(',','')
#             key = key.split(',')
#             key_0 = key[0].strip()
#             key_1 = key[1].strip()
#             id_0 = event_id[key_0]
#             id_1 = event_id[key_1]
#             node_set.update(key_0)
#             node_set.update(key_1)
#             edgelist.append((id_0, id_1, int(index)))
# G.add_weighted_edges_from(edgelist)
# for node in total_event:
#     if node not in node_set:
#         x = [(event_id[node], event_id[node],0)]
#         G.add_weighted_edges_from(x)

# print_graph_detail(G)


# edgelist = [(0, 1), (0, 2), (1, 3)]  # note that the order of edges
# G.add_edges_from(edgelist)
#
# # plot the graph
# fig, ax = plt.subplots(figsize=(4,4))
# option = {'font_family':'serif', 'font_size':'15', 'font_weight':'semibold'}
# nx.draw_networkx(G, node_size=400, **option)  #pos=nx.spring_layout(G)
# plt.axis('off')
# plt.show()
# nx.write_weighted_edgelist(G, f"./data/DuEE1.0/train_graph.txt")
# pos = nx.spiral_layout(G)
# nx.draw(G,pos)
# edge_labels = nx.get_edge_attributes(G,'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
# nx.draw_networkx_labels(G, pos, alpha=0.5)
# plt.rcParams['font.sans-serif'] = 'SimHei'
# plt.savefig('graph.png')
# plt.show()

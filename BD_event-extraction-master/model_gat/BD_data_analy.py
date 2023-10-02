import json
import torch
import sys
import os
sys.path.append(os.getcwd())
from BD_data_load import get_event_rep
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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


path_train = './data/train.json'
# path_dev = './data/dev.json'
# path_list = [path_train, path_dev]
event_id = {}
e_rep = 'event_name_seed'
event_rep = get_event_rep('./data/trigger_representation_DuEE.json', e_rep)
event_types = sorted(list(event_rep.keys()))
for index, value in enumerate(event_types):
    event_id[value] = index
parameter = []
for e in event_types:
    temp = event_rep[e].split('-')
    parameter.extend(temp[3:-1])
parameter = sorted(set(parameter))#216

for index, value in enumerate(parameter):
    event_id[value] = len(event_id)#281

   
events_cross = {}
node_set = set()
# for path in path_list:
with open(path_train, mode='r', encoding='utf-8') as f:
    for line in f.readlines():
        events= []
        data = json.loads(line)
        sentence = data['text']
        words = [word for word in sentence]
        if len(words) > 500:
            continue
        for event_mention in data['event_list']:
            if event_mention['event_type'] not in events:
                events.append(event_mention['event_type'])
        for i in range(len(events)):
            if i ==len(events)-1:
                continue
            for j in range(i+1,len(events)):
                key_0 = events[i]
                key_1 = events[j]
                node_set.add((key_0))
                node_set.add(key_1)
                id_0 = event_id[key_0]
                id_1 = event_id[key_1]
                if (id_0, id_1)not in events_cross.keys():
                    events_cross[(id_0, id_1)] = 1
                else:
                    events_cross[(id_0, id_1)] = events_cross[(id_0, id_1)] + 1

for e in event_types:
    temp =  event_rep[e].split('-')
    temp = temp[3:-1]
    id_0 = event_id[e]
    for x in temp:
        id_1 = event_id[x]
        events_cross[id_0, id_1] = 1

G = nx.Graph()
edgelist = []
for key, value in events_cross.items():
    edgelist.append((key[0],key[1],int(value)))
G.add_weighted_edges_from(edgelist)
for e in event_types:
    if e not in node_set:
        x = [(event_id[e], event_id[e],0)]
        G.add_weighted_edges_from(x)
print_graph_detail(G)
nx.write_weighted_edgelist(G, f"./model_gat/data/train_graph.txt")


            
# triggers = {}
# for trigger in events:
#     triggers[trigger.split('-')[-1]] = trigger
    
# file = './data/pred.json'
# result = []
# with open(file, mode='r', encoding='utf-8') as f:
#     name = json.load(f)
#     for data in name:
#         item = {}
#         item['id'] = data['id']
#         item["event_list"] = []
#         events = eval(data['event_list'])
       
#         text = data['words']
#         for trigger in events:
           
#             single_event = {}
#             st, ed, type = trigger
#             single_event['event_type'] = triggers[type]
#             single_event['arguments'] = []
#             for argument in events[trigger]:
#                 single_argument = {}
#                 ast, aed, atype = argument
#                 single_argument['role'] = atype
#                 single_argument['argument'] = text[ast:aed]
#                 single_event['arguments'].append(single_argument)
#             item["event_list"].append(single_event)
#         result.append(item)

# # solution 1 
# # jsObj = json.dumps(result)   
# with open('./data/pred_new.json', "w") as f:
#     for item in result:
#         jsObj = json.dumps(item, ensure_ascii=False, sort_keys=False)
#         f.write('{}\n'.format(jsObj))  
# f.close()
# for item in result:
#     with open('./data/pred_new.json', 'w') as f:
#         json.dump(result, f, indent=2, ensure_ascii=False,sort_keys=False)
# with open('./data/pred_new.json', 'w') as f:
#     json.dump(result, f, indent=2, ensure_ascii=False, sort_keys=False)
# file1 = './output_1/44_test'
# file2 = './data/test1.json'


# result = []
# with open(file1, mode='r', encoding='utf-8') as f:
#     events = []
#     for line in f.readlines():
#         if 'arguments' in line:
#             ev = line.split('#')[-1].strip()
#             events.append(ev)
# print(len(events))

# with open(file2, mode='r', encoding='utf-8') as f:
#     i = 0
#     for line in f.readlines():
#         item = {}
#         data = json.loads(line)
#         item['words'] = data['text']
#         item['id'] = data['id']
#         item['event_list'] = events[i]
#         result.append(item)
#         i = i + 1
        
# with open('./data/pred.json', 'w') as f:
#     json.dump(result, f, indent=2, ensure_ascii=False)
    
# from pytorch_pretrained_bert import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese-vocab.txt', do_lower_case=False)
# text = '一云南网红直播自己母亲葬礼，结果直播中突发意外，因为暴雨，雨棚倒塌，最终导致18人重伤2人轻伤，而主播在支付了部分医药费后，却玩起了失踪，甚至一度继续直播他人受伤，引发网友愤怒\n不少网友都不理解，直播母亲葬礼这个行为'
# text = text.replace('\n', '-')
# print(text)
# for t in text:
#     tokens = tokenizer.tokenize(t)
#     print(tokens)
#     tokens_xx = tokenizer.convert_tokens_to_ids(tokens)
#     print(tokens_xx)


# path_train = './data/train.json'
# path_dev = './data/dev.json'
# path_test = './data/test1.json'
# path_sample = './data/sample.json'
# path_list = [path_train, path_dev]
#
#
# events = []
# roles = []
# L = 0
# for path in path_list:
#     with open(path, mode='r', encoding='utf-8') as f:
#         for line in f.readlines():
#             data = json.loads(line)
#             sentence = data['text']
#             words = [word for word in sentence]
#             if len(words) > 500:
#                 print("===")
#
#             Entity = {}
#             for event_mention in data['event_list']:
#                 if event_mention['event_type'] not in events:
#                     events.append(event_mention['event_type'])
#                 for argument_mention in event_mention['arguments']:
#                     if argument_mention['role'] not in roles:
#                         roles.append(argument_mention['role'])
#                     if len(argument_mention['argument']) > L:
#                         L = len(argument_mention['argument'])
#                     if argument_mention['argument_start_index'] in Entity:
#                         if argument_mention['argument'] != Entity[argument_mention['argument_start_index']]:
#                             print(sentence)
#                     if argument_mention['argument_start_index'] not in Entity:
#                         Entity[argument_mention['argument_start_index']] =  argument_mention['argument']
#
#
# print(L)
# triggers = []
# for trigger in events:
#     triggers.append(trigger.split('-')[-1])
#
# if len(events) == len(set(triggers)):
#     print('OJBK')
# print(roles)
# print(len(roles))
# print(len(events))
# print(triggers)
# print(len(triggers))
#
#
# # with open('./data/num.json', 'w') as f:
# #     json.dump(b, f, indent=2, ensure_ascii=False)


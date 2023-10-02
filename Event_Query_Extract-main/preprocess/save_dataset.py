import sys
import os
sys.path.append("../")
sys.path.append("../../")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch
import argparse
from utils.data_to_dataloader import read_data_from
from utils.config import Config
from utils.metadata import Metadata
from collections import defaultdict
import spacy
from spacy.tokenizer import Tokenizer as spacy_tokenizer
from random import shuffle
from utils.data_to_dataloader import prepare_bert_sequence, pad_sequences, \
    bio_to_ids, firstSubwordsIdx_for_one_seq_template, prepare_sequence
import json
import numpy as np
from torch.utils.data import TensorDataset
import time


spacy_tagger = spacy.load("zh_core_web_lg")
spacy_tagger.tokenizer = spacy_tokenizer(spacy_tagger.vocab)


def token_to_berttokens(sent, tokenizer, omit=True, template=False):
    '''
    Generate Bert subtokens, and return first subword index list
    :param sent:
    :param tokenizer:
    :param omit:
    :param template:
    :return:
    '''
    bert_tokens = [tokenizer.tokenize(x) for x in sent]
    to_collect = [[1] + [0 for i in range(len(x[1:]))] for x in bert_tokens]
    if template:
        second_sep_idx = [i for i in range(len(sent)) if sent[i] == '[SEP]'][-1]
        bert_tokens_prefix = [tokenizer.tokenize(x) for x in sent[:second_sep_idx+1]]
        to_collect = [[1] + [0 for i in range(len(x[1:]))] for x in bert_tokens_prefix]
    bert_tokens = sum(bert_tokens, [])
    to_collect = sum(to_collect, [])
    if omit:
        omit_dic = {'[SEP]', '[CLS]'}
        # omit_dic = {'[CLS]'}
        to_collect = [x if y not in omit_dic else 0 for x, y in zip(to_collect, bert_tokens)]
    return bert_tokens, to_collect

def pair_trigger_template(data_bert, event_template, config):
    '''
    Pair trigger with event type query
    :param data_bert:
    :param event_template:
    :param config:
    :return:
    '''
    data_bert_new = []
    event_types = sorted(list(event_template.keys()))
    event_template_bert = {}
    for e in event_types:
        temp = event_template[e].split('-')
        event_template_bert[e] = token_to_berttokens(temp, config.tokenizer, template=True)#[0,[1,0],[1,0,0],0]

    for j in range(len(data_bert)):

        tokens, event_bio, arg_bio = data_bert[j]
        # print(tokens)
        if len(tokens) > 256:
            continue
        bert_tokens, to_collect = token_to_berttokens(tokens, config.tokenizer, False)
        pos_tag = get_pos(tokens)
        # print(event_bio)
        if set(event_bio[0]) == {'O'}:
            event_bio, arg_bio = [], []
        trigger_arg_dic = trigger_arg_bio_to_ids(event_bio, arg_bio, event_types, len(tokens))

        for event_type in event_types:
            this_template = event_template[event_type].split('-')
            this_template_bert, this_template_to_collect = event_template_bert[event_type]
            this_tokens = this_template + tokens + ['[SEP]']

            this_trigger_bio = [x[0] for x in trigger_arg_dic[event_type]]
            this_ner_arg_bio = [x[1] for x in trigger_arg_dic[event_type]]

            bert_sent = this_template_bert + bert_tokens[:] + ['[SEP]']

            sent_idx_to_collect = [0 for _ in range(len(this_template_bert))] + to_collect[:] + [0]
            data_tuple = (this_tokens, event_type, this_trigger_bio,
                          this_ner_arg_bio, pos_tag, bert_sent, this_template_to_collect, sent_idx_to_collect)

            data_bert_new.append(data_tuple)
    return data_bert_new


def trigger_arg_bio_to_ids(trigger_bio, arg_bio, event_type, sent_len):
    '''
    Convert list annotation to dictionary
    :param trigger_bio: Trigger list [[trigger_for_event_mention1], [trigger_for_event_mention2]]
    :param arg_bio: [[args_for_event_mention1], [args_for_event_mention2]]
    :param event_type: event type list
    :param sent_len:
    :return:
    '''
    ret = defaultdict(list)

    if trigger_bio:
        N = len(trigger_bio)
        for i in range(N):
            this_trigger = set(trigger_bio[i])
            this_trigger.remove('O')
            if this_trigger:
                this_trigger = list(this_trigger)[0][2:]
            ret[this_trigger].append([trigger_bio[i], arg_bio[i]])

    no_this_trigger = ['O'] * sent_len
    for i in event_type:
        if not ret[i]:
            ret[i] = [(no_this_trigger, no_this_trigger)]

    return ret

def arg_tags_to_ids(trigger_bio, arg_bio, event_type, arg_to_mappings):
    '''
    Convert list annotation to dictionary
    :param trigger_bio: Trigger list [[trigger_for_event_mention1], [trigger_for_event_mention2]]
    :param arg_bio: [[args_for_event_mention1], [args_for_event_mention2]]
    :param event_type: event type list
    :param sent_len:
    :return:
    '''
    ret = defaultdict(list)

    entity_mappings = []
    arg_tags = []
    # endids = []
    if trigger_bio:
        N = len(trigger_bio)
        for i in range(N):
            this_trigger = set(trigger_bio[i])
            this_trigger.remove('O')
            if this_trigger:
                this_trigger = list(this_trigger)[0][2:]
                arg_begins = [x for x in range(len(arg_bio[i])) if arg_bio[i][x][0] == 'B']
                arg_ends = []
                for a in arg_begins:
                    b = a+1
                    while b<len(arg_bio[i]):
                        if arg_bio[i][b][0] == 'I':
                            b+=1
                            continue
                        else:
                            break
                    arg_ends.append(b-1)
                num = len(entity_mappings) 
                for z, v in zip(arg_begins,arg_ends): 
                    entity_mapping = [0 for _ in range(len(arg_bio[0]))]
                    entity_mapping[z:v+1] =  [1 for _ in range(v-z+1)] 
                    entity_mappings.append(entity_mapping)
                paddings = [121] * num
                
                ret[this_trigger].append(paddings+[arg_to_mappings[arg_bio[i][args][2:]] if len(arg_bio[i][args]) > 2 else arg_to_mappings[arg_bio[i][args]] for args in arg_begins])
                    # ret[this_trigger].append(arg_tags)
    for key, value in ret.items():
        for x in range(len(value)):
            if len(ret[key][x])<len(entity_mappings):
                ret[key][x] = ret[key][x]+[121]*(len(entity_mappings)-len(ret[key][x]))
    print(ret)
                # time.slllep
    entity_mappings = pad_sequences(entity_mappings, maxlen=256, dtype="long", truncating="post", padding="post")

    no_this_arg = [121] 
    for i in event_type:
        if not ret[i]:
            ret[i].append(no_this_arg)

    return ret,entity_mappings


def get_pos(sentence):
    '''
    Get POS tag for input sentence
    :param sentence:
    :return:
    '''
    doc = spacy_tagger(' '.join(list(sentence)))
    ret = []
    for token in doc:
        ret.append(token.pos_)
    return ret


def remove_irrelevent_data(datas):
    '''
    Keep input sentence, trigger annotations and argument annotations
    :param data:
    :param ere:
    :return:
    '''
    to_collect_idx = [0, 2, 3]
    data = [[x[y] for y in to_collect_idx] for x in datas]
    for x in data:
        # if not ere:
        #     x[2] = x[2][::3]  # discard value and time arguments
        if x[1] == []:
            x[1] = [['O' for _ in range(len(x[0]))]]
            x[2] = [['O' for _ in range(len(x[0]))]]
    return data


def data_extract(data, shuffle=False):
    index_array = list(range(len(data)))
    if shuffle:
        np.random.shuffle(index_array)

    indices = index_array[:]
    data = [data[idx] for idx in indices]

    words_batch = [e[0] for e in data]
    event_type = [e[1] for e in data]
    event_bio = [e[2] for e in data]
    arg_bio = [e[3] for e in data]
    pos_tag = [e[4] for e in data]
    bert_tokens = [e[5] for e in data]
    idxs_to_collect_sent = [e[6] for e in data]
    idxs_to_collect_event = [e[7] for e in data]
    # entity_mapping = [e[8] for e in data]
    # arg_mapping = [e[9] for e in data]
    

    return words_batch, event_type, event_bio, arg_bio, pos_tag, bert_tokens, idxs_to_collect_sent, idxs_to_collect_event#entity_mapping, arg_mapping


def dataset_prepare_trigger_zero_template(data_bert, config, trigger_to_ids, metadata):
    """
    Generate data loader for the argument model
    :param data_bert:
    :param config:
    :param trigger_to_ids:
    :param metadata
    :return:
    """
    # unpack data
    word_to_ix = config.tokenizer.convert_tokens_to_ids

    tokens, event_type, event_bio, _, pos_tag, bert_sent, idxs_to_collect_event, idxs_to_collect_sent = data_extract(
        data_bert)
    # general information: sent_len, bert_sent_len, first_index
    bert_sentence_lengths = [len(s) for s in bert_sent]
    max_bert_seq_length = int(max(bert_sentence_lengths))
    sentence_lengths = [len(x) for x in tokens]
    max_seq_length = int(max(sentence_lengths))
    bert_tokens = prepare_bert_sequence(bert_sent, word_to_ix, config.PAD_TAG, max_bert_seq_length)
    # general information: pad_sequence
    idxs_to_collect_sent = pad_sequences(idxs_to_collect_sent, dtype="long", truncating="post", padding="post")
    idxs_to_collect_sent = torch.Tensor(idxs_to_collect_sent)
    idxs_to_collect_event = pad_sequences(idxs_to_collect_event, dtype="long", truncating="post", padding="post")
    idxs_to_collect_event = torch.Tensor(idxs_to_collect_event)

    sent_lengths = torch.Tensor(sentence_lengths).unsqueeze(1)
    pos_tags_all = prepare_sequence(pos_tag, metadata.pos2id, -1, max_seq_length)
    bert_sentence_lengths = torch.Tensor(bert_sentence_lengths)

    # trigger
    for i in range(len(event_bio)):
        if len(event_bio[i]) == 1:
            event_bio[i] = event_bio[i][0]
        else:
            event_bio[i] = [min(np.array(event_bio[i])[:, j]) for j in range(len(event_bio[i][0]))]
    event_tags = bio_to_ids(event_bio, trigger_to_ids, is_trigger=True)

    long_data = (sent_lengths, bert_sentence_lengths,
                 bert_tokens, idxs_to_collect_event, idxs_to_collect_sent, pos_tags_all, event_tags)
    return long_data

def pair_arg_template(data_bert, arg_template, config, arg_to_mappings):
    '''
    需要改
    Pair trigger with event type query
    :param data_bert:
    :param event_template:
    :param config:
    :return:
    '''
    data_bert_new = []
    event_types = sorted(list(arg_template.keys()))
    event_template_bert = {}
    for e in event_types:
        temp = ['[CLS]'] + arg_template[e] + ['[SEP]']
        event_template_bert[e] = token_to_berttokens(temp, config.tokenizer, template=True)

    for j in range(len(data_bert)):

        tokens, event_bio, arg_bio = data_bert[j]
        if len(tokens) > 256:
            continue
        bert_tokens, to_collect = token_to_berttokens(tokens, config.tokenizer, False)
        pos_tag = get_pos(tokens)
        if set(event_bio[0]) == {'O'}:
            event_bio, arg_bio = [], []
        trigger_arg_dic = trigger_arg_bio_to_ids(event_bio, arg_bio, event_types, len(tokens))
        arg_tags_dic, entity_mappings= arg_tags_to_ids(event_bio, arg_bio, event_types, arg_to_mappings)
        
        this_trigger1 = set()
        N = len(event_bio)
        for i in range(N):
            this_trigger = set(event_bio[i])
            this_trigger.remove('O')
            if this_trigger:
                if this_trigger not in this_trigger1:
                    this_trigger = list(this_trigger)[0][2:]
                    this_template = arg_template[this_trigger]
                    y = this_template + ['[SEP]']
                    arg_mapping = [arg_to_mappings[x] for x in y]
                    this_template_bert, this_template_to_collect = event_template_bert[this_trigger]
                    this_tokens = ['[CLS]']+ this_template + ['[SEP]'] + tokens + ['[SEP]']
                    bert_sent = this_template_bert + bert_tokens[:] + ['[SEP]']
                    sent_idx_to_collect = [0 for _ in range(len(this_template_bert))] + to_collect[:] + [0]
                    this_trigger_bio = trigger_arg_dic[this_trigger]
                    this_ner_arg_bio = arg_tags_dic[this_trigger]
                    for x in range(len(this_ner_arg_bio)):
                        arg = this_ner_arg_bio[x]
                        trigger = this_trigger_bio[x][0]
                        triggert_mask = [0 if x == 'O' else 1 for x in trigger]
                        data_tuple = (this_tokens, this_trigger, triggert_mask,
                                    arg, pos_tag, bert_sent, this_template_to_collect, sent_idx_to_collect, entity_mappings, arg_mapping)
                        data_bert_new.append(data_tuple)
                    this_trigger1.add(this_trigger)
                else:
                    continue

                    

        # for event_type in event_types:
        #     this_template = arg_template[event_type]
        #     y = this_template + ['[SEP]']
        #     arg_mapping = [arg_to_mappings[x] for x in y]
        #     this_template_bert, this_template_to_collect = event_template_bert[event_type]
        #     this_tokens = ['[CLS]']+ this_template + ['[SEP]'] + tokens + ['[SEP]']

        #     this_trigger_bio = [x[0] for x in trigger_arg_dic[event_type]]
        #     this_ner_arg_bio = arg_tags_dic[event_type][0]

        #     bert_sent = this_template_bert + bert_tokens[:] + ['[SEP]']

        #     sent_idx_to_collect = [0 for _ in range(len(this_template_bert))] + to_collect[:] + [0]
        #     data_tuple = (this_tokens, event_type, this_trigger_bio,
        #                   this_ner_arg_bio, pos_tag, bert_sent, this_template_to_collect, sent_idx_to_collect, entity_mappings, arg_mapping)

        #     data_bert_new.append(data_tuple)
    return data_bert_new

def dataset_prepare_arg_zero_template(data_bert, config, trigger_to_ids, arg_to_ids, metadata):
    """
    需要改！！
    Generate data loader for the argument model
    :param data_bert:
    :param config:
    :param trigger_to_ids:
    :param metadata
    :return:
    """
    # unpack data
    word_to_ix = config.tokenizer.convert_tokens_to_ids

    tokens, event_type, triggert_mask, arg_tags, pos_tag, bert_sent, idxs_to_collect_event, idxs_to_collect_sent, entity_mappings, arg_mapping = data_extract(
        data_bert)
    # general information: sent_len, bert_sent_len, first_index
    
    bert_sentence_lengths = [len(s) for s in bert_sent]
    max_bert_seq_length = int(max(bert_sentence_lengths))
    sentence_lengths = [len(x) for x in tokens]
    max_seq_length = int(max(sentence_lengths))
    bert_tokens = prepare_bert_sequence(bert_sent, word_to_ix, config.PAD_TAG, max_bert_seq_length)
    # general information: pad_sequence
    idxs_to_collect_sent = pad_sequences(idxs_to_collect_sent, dtype="long", truncating="post", padding="post")
    idxs_to_collect_sent = torch.Tensor(idxs_to_collect_sent)
    idxs_to_collect_event = pad_sequences(idxs_to_collect_event, dtype="long", truncating="post", padding="post")
    idxs_to_collect_event = torch.Tensor(idxs_to_collect_event)
    # print(max_seq_length)
    # print(np.array(entity_mapping).shape)
    # print(entity_mapping[1000])
    # arg_mask = pad_sequences(arg_mask, dtype="long", truncating="post", padding="post")
    entity_mappings_len = [len(s) for s in entity_mappings]
    max_entity_mappings_len = int(max(entity_mappings_len))
    entity_mappings, entity_num = prepare_entity_sequence(entity_mappings, 256, max_entity_mappings_len)
    
    # print(type(entity_mapping))
    arg_tags = pad_sequences(arg_tags, dtype="long", truncating="post", padding="post",value=122)
    arg_tags = torch.Tensor(arg_tags)
    
    
    sent_lengths = torch.Tensor(sentence_lengths).unsqueeze(1)
    pos_tags_all = prepare_sequence(pos_tag, metadata.pos2id, -1, max_seq_length)
    bert_sentence_lengths = torch.Tensor(bert_sentence_lengths)
    # x = 0
    # for i in range(len(event_bio)):
    #     if len(event_bio[i]) > 1:
    #         print(event_bio[i])
    #         x = x + 1
    # print(x)
    # time.sleep(10000)  
    # trigger
    # event_mask = []
    # # arg_mask = []
    # for i in range(len(event_bio)):
    #     if len(event_bio[i]) == 1:
    #         event_bio[i] = event_bio[i][0]
    #         event_mask.append([0 if x == 'O' else 1 for x in event_bio[i]])
    #         # arg_mask.append([0 if x == 'O' else 1 for x in arg_bio[i]])
    #         # print(event_bio)
    #         # time.sleep(10000)
    #     else:
    #         event_bio[i] = [min(np.array(event_bio[i])[:, j]) for j in range(len(event_bio[i][0]))]
    #         event_mask.append([0 if x == 'O' else 1 for x in event_bio[i]])
    #         # arg_mask.append([0 if x == 'O' else 1 for x in arg_bio[i]])
    #         # print(event_bio)
    #         # ids = [event_bio[i][0][j].startswith('B-')for j in range(len(event_bio[i][0]))]
    #         # event_bio[i] = np.array(event_bio[i][:, ids]
    #         # time.sleep(10000)
    # event_tags = bio_to_ids(event_bio, trigger_to_ids, is_trigger=True)
    
    event_mask = torch.Tensor(pad_sequences(triggert_mask, dtype="long", truncating="post", padding="post", value=0))
    # arg_mask = torch.Tensor(pad_sequences(arg_mask, dtype="long", truncating="post", padding="post", value=0))
    # arg_to_ids = torch.Tensor(arg_to_ids)
    # print(event_mask.shape)
    # print(bert_tokens.shape)
    arg_mapping = torch.Tensor(pad_sequences(arg_mapping, dtype="long", truncating="post", padding="post", value=arg_to_ids[config.PAD_TAG]))

    long_data = (sent_lengths, bert_sentence_lengths,
                 bert_tokens, idxs_to_collect_event, idxs_to_collect_sent, pos_tags_all, arg_tags, event_mask, entity_mappings,entity_num, arg_mapping)
    return long_data


def get_event_rep(f='../utils/trigger_representation_ace.json', rep='type_name_seed_template'):
    f = open(f, 'r')
    trigger_representation_json = json.load(f)
    f.close()
    return trigger_representation_json[rep]['suppliment_trigger']

def prepare_entity_sequence(seq_batch, emb_len, max_entity_num):
    padded_seqs = []
    entity_num = []
    for seq in seq_batch:
        entity_num.append(len(seq))
        y = []
        for x in seq:
            y.append(torch.tensor(x, dtype=torch.int))
        for _ in range(max_entity_num-len(seq)):
            pad_seq = torch.full((emb_len,), 0, dtype=torch.int)
        # ids = [to_ix(w) for w in seq]
        
            y.append(pad_seq)
        y = torch.stack(y)
        padded_seqs.append(y)
    padded_seqs = torch.tensor([item.detach().numpy() for item in padded_seqs])
    return padded_seqs, torch.tensor(entity_num, dtype=torch.int).unsqueeze(-1)


def save_trigger_dataset(dataset, path=None):
    dataset = [x.cuda() for x in dataset]
    # dataset = [x for x in dataset]
    tensor_set = TensorDataset(*dataset)
    if path:
        torch.save(tensor_set, path)
    print('save file to ', path)
    return 0


def save_to_json(data, file):
    res = []
    for x in data:
        event_list = []
        arg_list = []
        sentence, triggers, args = x
        sent_len = len(sentence)
        if set(triggers[0]) == {'O'}:
            res.append({'event_trigger': [], 'arg_list': []})
            continue
        for k in range(len(triggers)):
            # print(triggers)
            # print(sentence)
            # print(sent_len)
            trigger_ids = [i for i in range(sent_len) if triggers[k][i] != 'O']
            # print(trigger_ids)
            event_begin, event_end = trigger_ids[0], trigger_ids[-1] + 1
            event_type = triggers[k][event_begin][2:]
            arg_begins = [i for i in range(sent_len) if args[k][i][0] == 'B']
            arg_types = [args[k][i][2:] for i in range(sent_len) if args[k][i][0] == 'B']
            arg_ends = []
            for a in arg_begins:
                b = a+1
                while b<sent_len:
                    if args[k][b][0] == 'I':
                        b+=1
                        continue
                    else:
                        break
                arg_ends.append(b)
            arg_list.append([(event_type, x, y, z) for x, y, z in zip(arg_types, arg_begins, arg_ends)])
            event_list.append([event_type, event_begin, event_end])
        res.append({'event_trigger': event_list, 'arg_list': arg_list})
    # jsonString = json.dumps(res)
    # jsonFile = open(file, "w", encoding='utf-8')
    # jsonFile.write(jsonString)
    # jsonFile.close()
    with open(file,'w',encoding='utf-8')as f:
        json.dump(res, f,indent=2,ensure_ascii=False)
    print('save to ', file)
    return res


def read_from_source(args):
    config = Config()
    torch.backends.cudnn.enabled = False
    torch.manual_seed(39)

    metadata = Metadata()
    e_rep = 'event_name_seed'
    print('start')
    if args.ace:
        data_folder = args.data_folder
        trigger_to_ids = metadata.ace.triggers_to_ids
        event_rep = get_event_rep(config.project_root + './preprocess/ace/trigger_representation.json', e_rep)
        save_path = config.project_root + '/data/ace_en/pt/'
    elif args.ere:
        data_folder = args.data_folder
        trigger_to_ids = metadata.ere.triggers_to_ids
        event_rep = get_event_rep(config.project_root + './preprocess/ere/trigger_representation.json', e_rep)
        save_path = config.project_root + '/data/ere_en/pt/'
    elif args.DuEE:
        data_folder = args.data_folder
        trigger_to_ids = metadata.DuEE.triggers_to_ids
        args_to_ids = metadata.DuEE.args_to_ids
        ids_to_args = metadata.DuEE.ids_to_args
        args_to_mappings = metadata.DuEE.args_to_mappings
        mappongs_to_args = metadata.DuEE.mappings_to_args
        trigger_arg_dic = metadata.DuEE.trigger_arg_dic
        event_rep = get_event_rep(config.project_root + './preprocess/ace/trigger_representation_DuEE.json', e_rep)
        save_path = config.project_root + './data/DuEE1.0/pt/'
    # Read from source .txt file and write into .pt file
    for data_split in ['dev','train']:
        # data = open(file, 'r').read().split('\n\n')
        # output_all = [i.split('\n') for i in data]
        # output = []
        # if data_split == 'dev':
        #     output1 = []
        #     output2 = []
        #     file1 = './data/DuEE1.0/after-process/train_pad.doc2.txt'
        #     data1 = open(file1, 'r').read().split('\n\n')
        #     output_pad = [i.split('\n') for i in data1]
        #     for x in output_all:
        #         if len(x)>4:
        #             output1.append(x)
        #         else:
        #             output2.append(x)
        #     output1_choices = []
        #     output2_choices = []
        #     output1_choices = sample(output1,150)
        #     output2_choices = sample(output2,450)
        #     output_ =  output1_choices + output2_choices
        # else:
        #     output_ = output_all 
        if data_split == 'dev':
            path = data_folder + data_split + '.doc2.txt'
        else:
            path = data_folder + data_split + '.doc.txt'
        raw_data = read_data_from(path, config.tokenizer, data_split)
        data = remove_irrelevent_data(raw_data[:])
        print(len(data))

        save_to_json(data, save_path + data_split + '5.json')
        data_bert_train = pair_trigger_template(data, event_rep, config)
        # data_bertarg_train = pair_arg_template(data, trigger_arg_dic, config, args_to_mappings)
        d_loader = dataset_prepare_trigger_zero_template(data_bert_train, config, trigger_to_ids, metadata)
        # d_loader_arg = dataset_prepare_arg_zero_template(data_bertarg_train, config, trigger_to_ids, args_to_mappings, metadata)
        
        save_trigger_dataset(d_loader, save_path + data_split + '5.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ace', action='store_true')
    parser.add_argument('--ere', action='store_true')
    parser.add_argument('--DuEE', action='store_true')
    parser.add_argument('--data_folder', type=str,  required=True)
    args = parser.parse_args()
    assert args.ere or args.ace or args.DuEE is True, 'set either ACE or ERE with --ace or --ere'
    read_from_source(args)



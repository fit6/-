from torch.utils.data import DataLoader
import json
import torch
import time


def load_data_from_pt(dataset, batch_size, shuffle=False):
    """
    Load data saved in .pt
    :param dataset:
    :param batch_size:
    :param shuffle:
    :return:
    """
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)


def pack_data_to_trigger_model_joint(batch):
    """
    Prepare data for trigger detection model
    :param batch:
    :return:
    """
    # unpack and truncate data
    (_, bert_sentence_lengths,
     bert_tokens, idxs_to_collect_event, idxs_to_collect_sent, pos_tags, event_tags) = batch
    embedding_length = int(torch.max(bert_sentence_lengths).item())
    bert_sentence_in = bert_tokens[:, :embedding_length]
    idxs_to_collect_sent = idxs_to_collect_sent[:, :embedding_length]
    sent_lengths = torch.sum(idxs_to_collect_sent, dim=1).int()
    max_sent_len = int(torch.max(sent_lengths).item())
    trigger_tags = event_tags[:, :max_sent_len]
    pos_tags = pos_tags[:, :max_sent_len]

    # send data to gpu
    tmp = [bert_sentence_in, trigger_tags, idxs_to_collect_event, idxs_to_collect_sent, sent_lengths, \
           bert_sentence_lengths, pos_tags]
    return [x.long() for x in tmp] + [embedding_length]


def calculate_f1(gold, pred, tp, eps=1e-6):
    recall = tp / (gold + eps)
    precision = tp / (pred + eps)
    f1 = 2 * (recall * precision) / (recall + precision + eps)
    return f1, precision, recall


def pack_data_to_arg_model(batch):
    # unpack batch
    # (dataset_id, sent_idx_all, sent_lengths, bert_sentence_lengths,
    #  bert_tokens, first_subword_idxs, pred_indicator,
    #  arg_tags, arg_mask, arg_type_ids,
    #  pos_tags, entity_identifier, entity_tags, entity_mask, entity_mapping,
    #  trigger_feats) = batch
    (_, bert_sentence_lengths,
     bert_tokens, idxs_to_collect_event, idxs_to_collect_sent, pos_tags, arg_tags, trigger_mask, entity_mask, entity_num, arg_mapping) = batch

    embedding_length = int(torch.max(bert_sentence_lengths).item())
    bert_sentence_in = bert_tokens[:, :embedding_length]
    idxs_to_collect_sent = idxs_to_collect_sent[:, :embedding_length]
    sent_lengths = torch.sum(idxs_to_collect_sent, dim=1).int()

    max_sent_len = int(torch.max(sent_lengths).item())
    pos_tags = pos_tags[:, :max_sent_len]
    # arg_tags = arg_tags[:, :max_sent_len]
    entity_mask = entity_mask[:,:,:max_sent_len]
    # for i in range(entity_masks.shape[0]):
    #     entity_mask.append(entity_masks[i, :entity_num[i], :max_sent_len])
    # print(entity_mask[0])
    # print(arg_tags[0])
    # time.sleep(10000)


    # simple processing
    trigger_indicator = trigger_mask
    trigger_indicator = trigger_indicator[:, :max_sent_len]
    # max_bert_len = torch.max(bert_sentence_lengths)
    # max_sent_len = max([sum(first_subword_idxs[i] > 0) for i in range(len(first_subword_idxs))]) - 2

    # # to cuda
    # (dataset_id, sent_idx_all, sent_lengths, bert_sentence_lengths,
    #  bert_tokens, first_subword_idxs, pred_indicator, arg_tags,
    #  arg_type_ids,
    #  pos_tags, entity_identifier, entity_tags, entity_mask
    #  ) = tuple(map(torch.Tensor.long, (dataset_id, sent_idx_all, sent_lengths, bert_sentence_lengths,
    #                                    bert_tokens, first_subword_idxs, pred_indicator, arg_tags,
    #                                    arg_type_ids,
    #                                    pos_tags, entity_identifier, entity_tags, entity_mask
    #                                    )))
    # # slice1, one dimension data
    # dataset_id = dataset_id[:max_bert_len]
    # sent_idx_all = sent_idx_all[:max_bert_len]
    # bert_sentence_lengths = bert_sentence_lengths[:max_bert_len]
    # # slice2, 2d, bs x bert_len
    # bert_tokens = bert_tokens[:, :max_bert_len]
    # # slice3, 2d, bs x sent_len
    # first_subword_idxs = first_subword_idxs[:, :max_sent_len + 2]
    # trigger_indicator = trigger_indicator[:, :max_sent_len]
    # arg_tags = arg_tags[:, :max_sent_len]
    # arg_type_ids = arg_type_ids[:, :max_sent_len]
    # pos_tags = pos_tags[:, :max_sent_len]
    # entity_identifier = entity_identifier[:, :max_sent_len]
    # entity_tags = entity_tags[:, :max_sent_len]
    # entity_mask = entity_mask[:, :max_sent_len]
    # entity_mapping = entity_mapping[:, :max_sent_len]
    # first_subword_idxs -= 2
    # return bert_sentence_lengths, bert_sentence_in, idxs_to_collect_event, idxs_to_collect_sent, \
    #        trigger_indicator, arg_tags, arg_mask, arg_type_ids, \
    #        pos_tags, entity_identifier, entity_tags, entity_mask, entity_mapping, trigger_feats
    temp = [bert_sentence_lengths, bert_sentence_in, idxs_to_collect_event, idxs_to_collect_sent, \
           trigger_indicator, arg_tags, pos_tags,entity_mask, arg_mapping, entity_num]

    return [x.long() for x in temp]


def save_to_json(json_output, path):
    jsonString = json.dumps(json_output)
    jsonFile = open(path, "w")
    jsonFile.write(jsonString)
    jsonFile.close()
    return


def save_to_jsonl(json_output, path):
    with open(path, 'w',encoding='utf-8') as outfile:
        for entry in json_output:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')
    outfile.close()
    return


def load_from_jsonl(json_path):
    data = []
    with open(json_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def pred_to_event_mention(pred, ids_to_triggers, config):
    ret = []
    for i in range(config.event_count):

        if not torch.any(pred[i]>0.5):
            continue

        temp = torch.cat([torch.tensor([0]).cuda(), pred[i], torch.tensor([0]).cuda()])
        is_event, begin, end = 0, None, None
        for j in range(len(temp)):
            if temp[j] and not is_event:
                begin = j-1
                is_event = 1
            if not temp[j] and is_event:
                end = j-1
                is_event = 0
                ret.append(tuple([ids_to_triggers[i], begin, end]))
    return ret

def pred_to_arg_mention(pred, ids_to_args, config, gold_event,entity_mappings, trigger_indicator_all):
    ret = []
    ids = 0
    tp, pos, gold = 0, 0, 0
    for i in range(len(gold_event)):
        event_num = len(gold_event[i]['event_trigger'])
        event_args = []
        for index in range(event_num):
            temps = pred[ids]
            entity_mapping = entity_mappings[ids]
            args = []
            for x in range(len(entity_mapping)):
                is_entity, begin, end = 0, None, None
                entity = entity_mapping[x]
                if torch.any(entity!=0):
                    temp = int(temps[x])
                    if temp < config.arg_count:
                        for j in range(len(entity)):
                            if entity[j] and not is_entity:
                                begin = j
                                is_entity = 1
                            if not entity[j] and is_entity:
                                end = j
                                is_entity = 0
                                args.append(tuple([gold_event[i]['event_trigger'][index][0], ids_to_args[temp], begin, end]))
            
            
            ids = ids + 1
            event_args.append(args) 
        
              
        this_pred = (set(tuple(x) for x in event_args))
        
        this_gold = set(tuple(tuple(y)for y in x) for x in gold_event[i]['arg_list'])
        # for ele in this_gold:
        #     print(ele)
        #     time.sleep(11000)
        tp += len(this_gold.intersection(this_pred))
        pos += len(this_pred)
        gold += len(this_gold)
        ret.append({"pred":event_args})

    return ret, tp, pos, gold
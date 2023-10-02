from BD_data_analy import print_graph_detail
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
import sys
import os
from pytorch_pretrained_bert import BertModel
sys.path.append(os.getcwd())
from BD_data_load import get_event_rep
from model_gat.data.config import Config
import json

def prepare_bert_sequence(seq_batch, to_ix, pad, emb_len):
    padded_seqs = []
    for seq in seq_batch:
        pad_seq = th.full((emb_len,), to_ix(pad), dtype=th.int)
        # ids = [to_ix(w) for w in seq]
        ids = to_ix(seq)
        pad_seq[:len(ids)] = th.tensor(ids, dtype=th.long)
        padded_seqs.append(pad_seq)
    return th.stack(padded_seqs)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj, is_sparse=False):
    """Preprocessing of adjacency matrix for simple pygGCN model and conversion to
    tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    if is_sparse:
        adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
        return adj_normalized
    else:
        return th.from_numpy(adj_normalized.A).float()
def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class PrepareData:
    def __init__(self, path):
        print("prepare data")
        config = Config()
        self.graph_path = "./model_gat/data"
        total_event = []
        event_id = {}
        tokenizer = config.tokenizer
        word_to_ix = tokenizer.convert_tokens_to_ids
        bert = BertModel.from_pretrained('./data/chinese_wwm_ext_pytorch')
        scheme_path = './event_schema/event_schema.json'
        with open(scheme_path, 'r', encoding='utf-8') as j:
            for line in j:
                context = json.loads(line)
                total_event.append(context['event_type'])
        for index, event_type in enumerate(total_event):
            event_id[event_type] = index
        # graph
        graph = nx.read_weighted_edgelist(f"{self.graph_path}/{path}.txt"
                                          , nodetype=str)
        print_graph_detail(graph)
        # adj = nx.to_scipy_sparse_matrix(graph)
        adj = nx.to_scipy_sparse_array(graph)
        
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        
        #
        adj = preprocess_adj(adj, is_sparse=True)

        #
        # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #
        # # features
        e_rep = "event_name_seed"
        event_rep = get_event_rep(config.project_root + './data/trigger_representation_DuEE.json', e_rep)
        event_types = sorted(list(event_rep.keys()))
        parameter = []
        for e in event_types:
            temp = event_rep[e].split('-')
            parameter.extend(temp[3:-1])
        parameter = sorted(set(parameter))#216
        parameter_token = [tokenizer.tokenize(x) for x in parameter]
        event_token = [tokenizer.tokenize(x) for x in event_types] + parameter_token
        bert_sentence_lengths = [len(s) for s in event_token]
        max_bert_seq_length = int(max(bert_sentence_lengths))
        # max_bert_seq_length = 768
        event_bert_tokens = prepare_bert_sequence(event_token, word_to_ix, config.PAD_TAG, max_bert_seq_length)
        encoded_layers, _ = bert(input_ids=event_bert_tokens)
        event_bert_tokens_embeding = encoded_layers[-1]
        event_bert_tokens_embeding = event_bert_tokens_embeding.detach()
        event_bert_tokens_embeding =  np.array(event_bert_tokens_embeding.sum(1), dtype=np.float32)/event_bert_tokens_embeding.shape[1]
        event_features = normalize_features(event_bert_tokens_embeding)
        self.adj = th.FloatTensor(np.array(adj.to_dense()))
        
        self.event_features = th.FloatTensor(np.array(event_features))


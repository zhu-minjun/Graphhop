import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import transformers
transformers.logging.set_verbosity_error()
import pandas as pd
import numpy as np
import torch
from accelerate import Accelerator
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import os
import time
from torch.autograd import Variable
from transformers import BertTokenizer,AutoTokenizer,AdamW,get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score
import argparse
accelerator = Accelerator()
device = accelerator.device

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0)
parser.add_argument("--deepspeed_config")

args = parser.parse_args()
CFG = {
    'seed': 2,
    'model': r'bert-base-uncased',
    'max_len': 30,
    'epochs': 10,
    'train_bs': 4,
    'valid_bs': 24,
    'lr': 1e-5,
    'num_workers': 0,
    'accum_iter': 1,
    'weight_decay': 1e-4,
    'device': 0,
    'ues_r_drop':True,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['seed'])

torch.cuda.set_device(CFG['device'])

def read_NDB(data_file):
    with open(data_file,encoding='utf-8') as file:
        text = json.load(file)
        return text

train_path = r'datas/train.json'
valid_path = r'datas/test.json'
test_path = r'datas/test.json'

train = read_NDB(train_path)
valid = read_NDB(valid_path)
test = read_NDB(test_path)

tokenizer = AutoTokenizer.from_pretrained(CFG['model'])



class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        data = self.df[idx]

        return data
import networkx as nx

class TaxStruct(nx.DiGraph):
    def __init__(self, edges):
        super().__init__(edges)
        self._root = ''
        for node in self.nodes:
            if self.in_degree(node) == 0:
                self._root = node
                break
        self.leaf_node = [node for node in self.nodes.keys() if self.out_degree(node) == 0]
        self.nearroot_node = [i[1] for i in edges if i[0] == 'Q']
        self.node2path_F = dict()
        self.node2path_B = dict()
        for node in self.nodes.keys():
            if node != self._root:
                nF = []
                nF_node = [i[1] for i in edges if i[0] == node]
                for path in nx.all_simple_paths(self, source=self._root, target=node):
                    nF.append([path,nF_node])

                nB = []
                nB_node = [i[0] for i in edges if i[1] == node]
                if node in self.leaf_node:
                    nB.append([[node],nB_node])
                else:
                    for leaf_node in self.leaf_node:
                        for path in nx.all_simple_paths(self, source=node, target=leaf_node):
                            nB.append([list(reversed(path)),nB_node])


                self.node2path_F[node] = nF
                self.node2path_B[node] = nB

def collate_fn(data):
    input_ids_question, attention_mask_question, input_ids_passage, attention_mask_passage,trees,evidence_dict,evidence_dict_EN = [], [], [],[],[],{},{}
    num = 0
    for it in range(len(data)):
        x = data[it]
        text = tokenizer(x['question'],padding='max_length', truncation=True, max_length=CFG['max_len'])

        input_ids_question.append(text['input_ids'])
        attention_mask_question.append(text['attention_mask'])

        replace_df = []
        for df in x['fact']:


            text = tokenizer(df, padding='max_length', truncation=True, max_length=CFG['max_len'])
            input_ids_passage.append(text['input_ids'])
            attention_mask_passage.append(text['attention_mask'])
            df_o = df
            while df in evidence_dict:
                df += '_'
            replace_df.append([df_o, df])
            evidence_dict[df] = num
            evidence_dict_EN[num] = df
            num += 1
        tax_list = []
        for its in x['order_true_fact']:
            if its['relation']['after'] != []:
                for after in its['relation']['after']:
                    facts = its['facts']
                    for d in replace_df:
                        facts = facts.replace(d[0],d[1])
                        after = after.replace(d[0],d[1])
                    tax_list.append((facts, after))
            if its['relation']['before'] == []:
                facts = its['facts']
                for d in replace_df:
                    facts = facts.replace(d[0], d[1])
                tax_list.append(('Q', facts))
        tax = TaxStruct(tax_list)
        trees.append(tax)


    input_ids_question = torch.tensor(input_ids_question,device=device)
    attention_mask_question = torch.tensor(attention_mask_question ,device=device)
    input_ids_passage = torch.tensor(input_ids_passage,device=device)
    attention_mask_passage = torch.tensor(attention_mask_passage ,device=device)
    return input_ids_question, attention_mask_question,input_ids_passage,attention_mask_passage,trees,evidence_dict,evidence_dict_EN

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_model(model, train_loader,epoch):
    model.eval()
    outputs = []

    model.train()

    losses = AverageMeter()
    optimizer.zero_grad()
    tk = tqdm(train_loader, total=len(train_loader), position=0,desc='Train')
    for step, (input_ids_question, attention_mask_question,input_ids_passage,attention_mask_passage,trees,evidence_dict,evidence_dict_EN) in enumerate(tk):
        loss = model(input_ids_question=input_ids_question,attention_mask_question=attention_mask_question,
                                                                 input_ids_passage=input_ids_passage,attention_mask_passage=attention_mask_passage,
                                                                 trees=trees,evidence_dict=evidence_dict)
        scaler.scale(loss).backward()
        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        losses.update(loss.item())
        tk.set_postfix(loss=losses.avg)

    return losses.avg

def test_model(model_F,model_B, val_loader):
    model_F.eval()
    model_B.eval()
    trees = []
    preds_F = []
    preds_B = []
    tk = tqdm(val_loader,desc='Test')
    with torch.no_grad():
        for step, (input_ids_question, attention_mask_question, input_ids_passage, attention_mask_passage,tree,evidence_dict,evidence_dict_EN) in enumerate(tk):
            trees.extend(tree)
            preds_B.extend(model_B.generate(input_ids_question=input_ids_question, attention_mask_question=attention_mask_question,
                         input_ids_passage=input_ids_passage, attention_mask_passage=attention_mask_passage,evidence_dict=evidence_dict,
                                    evidence_dict_EN=evidence_dict_EN))
            preds_F.extend(model_F.generate(input_ids_question=input_ids_question, attention_mask_question=attention_mask_question,
                         input_ids_passage=input_ids_passage, attention_mask_passage=attention_mask_passage,evidence_dict=evidence_dict,
                                    evidence_dict_EN=evidence_dict_EN))

    return preds_F,preds_B,trees


train_set = MyDataset(train)
valid_set = MyDataset(valid)
test_set = MyDataset(test)

from Model_F import TreeHot_F
from Model_B import TreeHot_B


scaler = GradScaler()

criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                              num_workers=CFG['num_workers'])
valid_loader = DataLoader(valid_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                              num_workers=CFG['num_workers'])
test_loader = DataLoader(test_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                              num_workers=CFG['num_workers'])

from scipy.optimize import minimize
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
def optimal_threshold(y_true, y_pred):
    loss = lambda t: -np.mean(f1_score(y_true,y_pred>np.tanh(t)))
    result = minimize(loss, 1, method='Powell')
    return np.tanh(result.x), -result.fun
loss = lambda t: -np.mean((np.array(all_truth) > 0.5) == (np.array(y_pred) > np.tanh(t)))

def Levenshtein_Distance(str1, str2):

    matrix = [[ i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if(str1[i-1] == str2[j-1]):
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)

    return matrix[len(str1)][len(str2)]
def log(text,path):
    with open(path+'/log.txt','a',encoding='utf-8') as f:
        f.write('-----------------{}-----------------'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        for i in text:
            f.write(i)
            print(i)
            f.write('\n')
        f.write('\n')
import os
def log_start(log_name):
    if log_name == '':
        log_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    try:
        os.mkdir('log/' + log_name)
    except:
        log_name += time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        os.mkdir('log/' + log_name)

    with open('log/' + log_name + '/python_file.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__),'r',encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)
    with open('log/' + log_name + '.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__),'r',encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)
    path = 'log/' + log_name
    with open(path+'/log.txt', 'a', encoding='utf-8') as f:
        f.write(log_name)
        f.write('\n')
    return path


log_name = 'TreeHot_eval_test'
path = log_start(log_name)


model_F = TreeHot_F.from_pretrained(r'./log/TreeHot_Forward/19').to(device)
model_B = TreeHot_B.from_pretrained(r'./log/TreeHot_Backward/19').to(device)
pred_F,pred_B,tree = test_model(model_F,model_B,valid_loader)
f1scores,pscores,rscores = [],[],[]
structural_acc,geds,graph_acc,EM = [],[],[],[]
pd.to_pickle(pred_F,path+'/pred_F.pk')
pd.to_pickle(pred_B,path+'/pred_B.pk')
pd.to_pickle(tree,path+'/tree.pk')
from tqdm import trange
program = {'Query':['What','QueryAttr','QueryRelation'],
            'Comparison':['SelectAmong','SelectBetween'],
            'Count':['Count'],
            'Bool':['VerifyNum','VerifyStr','VerifyYear'],
            'Qualifier':['QueryAttrQualifie','QueryRelationQualifier']}


n = 0.2

for i in trange(len(tree)):
    pred_m = np.zeros([len(valid[i]['fact'])])
    label_m = np.zeros([len(valid[i]['fact'])])

    edge = tree[i].edges
    edge_item = []
    for p in edge:
        edge_item.append(str([p[0],p[1]]))
    ps_edge1,ps_edge2 = [],[]
    for p in pred_F[i]:
        ps_edge1.append(str(p))
    for p in pred_B[i]:
        ps_edge2.append(str(p))
    ps_edge = list(set(ps_edge1)|set(ps_edge2))

    ps_graph = []
    if  (len(set(ps_edge1) & set(ps_edge2))) / (len(set(ps_edge1) | set(ps_edge2))+ 0.0001) > (n/10):
        for p in list(set(ps_edge2)):
            ps_graph.append(eval(p))
    else:
        ps_graph_str = list(set(ps_edge1) | set(ps_edge2))
        for p in ps_graph_str:
            gra = TaxStruct(ps_graph)
            if eval(p)[1] not in gra.nodes.keys():
                ps_graph.append(eval(p))
        for p in ps_graph_str:
            if eval(p)[1] in gra.leaf_node:
                ps_graph.append(eval(p))

    graph = nx.DiGraph(ps_graph)
    ps_graphstr = [str(x) for x in ps_graph]
    if set(ps_graphstr) == set(edge_item):
        structural_acc.append(1)
    else:
        structural_acc.append(0)
    GM = nx.isomorphism.GraphMatcher(graph, tree[i])
    if GM.is_isomorphic():
        graph_acc.append(1)
    else:
        graph_acc.append(0)
    ps = list(graph.nodes.keys())
    ts = []
    for k in list(tree[i].nodes.keys()):
        if k != 'Q':
            ts.append(k)
    for t_item in range(len(valid[i]['fact'])):
        if valid[i]['fact'][t_item] in ps:
            pred_m[t_item] = 1
        if valid[i]['fact'][t_item] in ts:
            label_m[t_item] = 1
    f1s = f1_score(pred_m, label_m, average='binary')
    p_score = precision_score(pred_m,label_m)
    r_score = recall_score(pred_m,label_m)
    if sum(pred_m == label_m)==len(pred_m):
        EM.append(1)
    else:
        EM.append(0)
    f1scores.append(f1s)
    pscores.append(p_score)
    rscores.append(r_score)
    geds.append(nx.graph_edit_distance(graph,tree[i],timeout=0.8))
    #geds.append(1)

print('F1:{} P:{} R:{} EM:{} Structural Acc:{} Graph Acc:{} GED:{}'.format(sum(f1scores)/len(f1scores),sum(pscores)/len(pscores),sum(rscores)/len(rscores),sum(EM)/len(EM),sum(structural_acc)/len(structural_acc),sum(graph_acc)/len(graph_acc),sum(geds)/len(geds)))

"""
for kp,vp in program.items():
    f1scores, pscores, rscores = [], [], []
    structural_acc, geds, graph_acc, EM = [], [], [], []
    num = 0
    count = 0
    for i in trange(len(tree)):
        if valid[i]['program'][-1]['function'] not in vp:
            continue
        if valid[i]['program'][-1]['function'] == 'Count':
            count = i
        num+=1
        pred_m = np.zeros([len(valid[i]['fact'])])
        label_m = np.zeros([len(valid[i]['fact'])])

        edge = tree[i].edges
        edge_item = []
        for p in edge:
            edge_item.append(str([p[0],p[1]]))
        ps_edge1,ps_edge2 = [],[]
        for p in pred_F[i]:
            ps_edge1.append(str(p))
        for p in pred_B[i]:
            ps_edge2.append(str(p))
        ps_edge = list(set(ps_edge1)|set(ps_edge2))

        ps_graph = []
        if len(set(ps_edge1) | set(ps_edge2)) / (len(set(ps_edge1) & set(ps_edge2)) + 0.0001) < 5:
            for p in list(set(ps_edge2)):
                ps_graph.append(eval(p))
        else:
            ps_graph_str = list(set(ps_edge1) | set(ps_edge2))
            for p in ps_graph_str:
                gra = TaxStruct(ps_graph)
                if eval(p)[1] not in gra.nodes.keys():
                    ps_graph.append(eval(p))
            for p in ps_graph_str:
                if eval(p)[1] in gra.leaf_node:
                    ps_graph.append(eval(p))

        graph = nx.DiGraph(ps_graph)
        ps_graphstr = [str(x) for x in ps_graph]
        if set(ps_graphstr) == set(edge_item):
            structural_acc.append(1)
        else:
            structural_acc.append(0)
        GM = nx.isomorphism.GraphMatcher(graph, tree[i])
        if GM.is_isomorphic():
            graph_acc.append(1)
        else:
            graph_acc.append(0)
        ps = list(graph.nodes.keys())
        ts = []
        for k in list(tree[i].nodes.keys()):
            if k != 'Q':
                ts.append(k)
        for t_item in range(len(valid[i]['fact'])):
            if valid[i]['fact'][t_item] in ps:
                pred_m[t_item] = 1
            if valid[i]['fact'][t_item] in ts:
                label_m[t_item] = 1
        f1s = f1_score(pred_m, label_m, average='binary')
        p_score = precision_score(pred_m,label_m)
        r_score = recall_score(pred_m,label_m)
        if sum(pred_m == label_m)==len(pred_m):
            EM.append(1)
        else:
            EM.append(0)
        f1scores.append(f1s)
        pscores.append(p_score)
        rscores.append(r_score)
        #geds.append(nx.graph_edit_distance(graph,tree[i],timeout=0.8))
        geds.append(1)


    print('{} {}  F1:{} P:{} R:{} EM:{} Structural Acc:{} Graph Acc:{} GED:{}'.format(kp,num,sum(f1scores)/len(f1scores),sum(pscores)/len(pscores),sum(rscores)/len(rscores),sum(EM)/len(EM),sum(structural_acc)/len(structural_acc),sum(graph_acc)/len(graph_acc),sum(geds)/len(geds)))


for na in ['scsh','mcsh','scmh','mcmh']:
    f1scores, pscores, rscores = [], [], []
    structural_acc, geds, graph_acc, EM = [], [], [], []
    for i in trange(len(tree)):
        pred_m = np.zeros([len(valid[i]['fact'])])
        label_m = np.zeros([len(valid[i]['fact'])])

        tax = tree[i]
        types = 'scsh'
        path_deep = 0
        path_breadth = len(tax.nearroot_node)
        for paths in tax.node2path_F.values():
            for p in paths:
                if len(p[0]) > path_deep:
                    path_deep = len(p[0])
                if len(p[1]) > path_breadth:
                    path_breadth = len(p[1])
        if path_deep < 3 and path_breadth < 2:
            types = 'scsh'
        if path_deep >= 3 and path_breadth < 2:
            types = 'mcsh'
        if path_deep < 3 and path_breadth >= 2:
            types = 'scmh'
        if path_deep >= 3 and path_breadth >= 2:
            types = 'mcmh'

        if types == na:

            edge = tree[i].edges
            edge_item = []
            for p in edge:
                edge_item.append(str([p[0],p[1]]))
            ps_edge1,ps_edge2 = [],[]
            for p in pred_F[i]:
                ps_edge1.append(str(p))
            for p in pred_B[i]:
                ps_edge2.append(str(p))
            ps_edge = list(set(ps_edge1)|set(ps_edge2))

            ps_graph = []
            if  (len(set(ps_edge1) & set(ps_edge2))) / (len(set(ps_edge1) | set(ps_edge2))+ 0.0001) > (n/10):
                for p in list(set(ps_edge2)):
                    ps_graph.append(eval(p))
            else:
                ps_graph_str = list(set(ps_edge1) | set(ps_edge2))
                for p in ps_graph_str:
                    gra = TaxStruct(ps_graph)
                    if eval(p)[1] not in gra.nodes.keys():
                        ps_graph.append(eval(p))
                for p in ps_graph_str:
                    if eval(p)[1] in gra.leaf_node:
                        ps_graph.append(eval(p))

            graph = nx.DiGraph(ps_graph)
            ps_graphstr = [str(x) for x in ps_graph]
            if set(ps_graphstr) == set(edge_item):
                structural_acc.append(1)
            else:
                structural_acc.append(0)
            GM = nx.isomorphism.GraphMatcher(graph, tree[i])
            if GM.is_isomorphic():
                graph_acc.append(1)
            else:
                graph_acc.append(0)
            ps = list(graph.nodes.keys())
            ts = []
            for k in list(tree[i].nodes.keys()):
                if k != 'Q':
                    ts.append(k)
            for t_item in range(len(valid[i]['fact'])):
                if valid[i]['fact'][t_item] in ps:
                    pred_m[t_item] = 1
                if valid[i]['fact'][t_item] in ts:
                    label_m[t_item] = 1
            f1s = f1_score(pred_m, label_m, average='binary')
            p_score = precision_score(pred_m,label_m)
            r_score = recall_score(pred_m,label_m)
            if sum(pred_m == label_m)==len(pred_m):
                EM.append(1)
            else:
                EM.append(0)
            f1scores.append(f1s)
            pscores.append(p_score)
            rscores.append(r_score)
            geds.append(nx.graph_edit_distance(graph,tree[i],timeout=1))
            #geds.append(1)

    with open('chain.txt','a',encoding='utf-8') as f:
        f.write('{}  F1:{} P:{} R:{} EM:{} Structural Acc:{} Graph Acc:{} GED:{} \n'.format('双向解耦_{}'.format(na),sum(f1scores)/len(f1scores),sum(pscores)/len(pscores),sum(rscores)/len(rscores),sum(EM)/len(EM),sum(structural_acc)/len(structural_acc),sum(graph_acc)/len(graph_acc),sum(geds)/len(geds)))

print(1)

with open('QA_output/predBTR_Coupling_bi_25.json') as f:
    pred = json.load(f)

for na in ['scsh','mcsh','scmh','mcmh']:
    f1scores, pscores, rscores = [], [], []
    structural_acc, geds, graph_acc, EM = [], [], [], []
    QAEM = []
    for i in trange(len(tree)):
        pred_m = np.zeros([len(valid[i]['fact'])])
        label_m = np.zeros([len(valid[i]['fact'])])

        tax = tree[i]
        types = 'scsh'
        path_deep = 0
        path_breadth = len(tax.nearroot_node)
        for paths in tax.node2path_F.values():
            for p in paths:
                if len(p[0]) > path_deep:
                    path_deep = len(p[0])
                if len(p[1]) > path_breadth:
                    path_breadth = len(p[1])
        if path_deep < 3 and path_breadth < 2:
            types = 'scsh'
        if path_deep >= 3 and path_breadth < 2:
            types = 'mcsh'
        if path_deep < 3 and path_breadth >= 2:
            types = 'scmh'
        if path_deep >= 3 and path_breadth >= 2:
            types = 'mcmh'

        if types == na:
            QAEM.append(pred[i]['pred_label'])
    print('{} {}'.format(na,sum(QAEM)/len(QAEM)))
"""
import os
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
    'seed': 42,
    'model': r'bert-base-uncased',
    'max_len': 30,
    'epochs': 20,
    'train_bs': 8,
    'valid_bs': 24,
    'lr': 1e-5,
    'num_workers': 0,
    'accum_iter': 1,
    'weight_decay': 1e-4,
    'device': 1,
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
valid_path = r'datas/valid.json'
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
        assert self._root == 'Q'
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

def valid_model(model, val_loader):
    model.eval()
    losses = AverageMeter()
    preds = []
    nearroot_labels,nearroot_preds,stop_labels, stop_preds,halfway_labels, halfway_preds = [],[],[],[],[],[]
    tk = tqdm(val_loader,desc='Valid')
    with torch.no_grad():
        for step, (input_ids_question, attention_mask_question, input_ids_passage, attention_mask_passage,trees,evidence_dict,evidence_dict_EN) in enumerate(tk):
            loss,nearroot_label,nearroot_pred,stop_label, stop_pred,halfway_label, halfway_pred = model.forward_valid(input_ids_question=input_ids_question, attention_mask_question=attention_mask_question,
                         input_ids_passage=input_ids_passage, attention_mask_passage=attention_mask_passage,
                         trees=trees, evidence_dict=evidence_dict)
            nearroot_labels.extend(nearroot_label)
            nearroot_preds.extend(nearroot_pred)
            stop_labels.extend(stop_label)
            stop_preds.extend(stop_pred)
            halfway_labels.extend(halfway_label)
            halfway_preds.extend(halfway_pred)
            losses.update(loss.item())
            tk.set_postfix(loss=losses.avg)

    return nearroot_labels,nearroot_preds,stop_labels, stop_preds,halfway_labels, halfway_preds
def test_model(model, val_loader,near_threshold,stop_threshold,halfway_threshold):
    model.eval()
    trees = []
    preds = []
    tk = tqdm(val_loader,desc='Test')
    with torch.no_grad():
        for step, (input_ids_question, attention_mask_question, input_ids_passage, attention_mask_passage,tree,evidence_dict,evidence_dict_EN) in enumerate(tk):
            output = model.generate(input_ids_question=input_ids_question, attention_mask_question=attention_mask_question,
                         input_ids_passage=input_ids_passage, attention_mask_passage=attention_mask_passage,evidence_dict=evidence_dict,
                                    evidence_dict_EN=evidence_dict_EN,near_threshold=near_threshold,stop_threshold=stop_threshold,halfway_threshold=halfway_threshold)
            trees.extend(tree)
            preds.extend(output)

    return preds,trees



train_set = MyDataset(train)
valid_set = MyDataset(valid)
test_set = MyDataset(test)

from Model_F import TreeHot_F
model = TreeHot_F.from_pretrained(CFG['model']).to(device)


scaler = GradScaler()
optimizer = AdamW(model.parameters(),lr=CFG['lr'], weight_decay=CFG['weight_decay'])
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                              num_workers=CFG['num_workers'])
valid_loader = DataLoader(valid_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                              num_workers=CFG['num_workers'])
test_loader = DataLoader(test_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                              num_workers=CFG['num_workers'])

scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                            CFG['epochs'] * len(train_loader) // CFG['accum_iter'])

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


log_name = 'TreeHot_Forward'
path = log_start(log_name)

BEST_score = 0.
BEST_test_score = None
for epoch in range(CFG['epochs']):
    train_loss = train_model(model, train_loader,epoch)
    model.save_pretrained(path+'/{}'.format(epoch))
    tokenizer.save_pretrained(path+'/{}'.format(epoch))

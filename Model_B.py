from transformers import RobertaModel,RobertaPreTrainedModel,BertModel,BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertAttention, BertSelfAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,  #B num_generated_triples H
        encoder_hidden_states,
        encoder_attention_mask
    ):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0] #hidden_states.shape
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :] # B 1 1 H
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0 #1 1 0 0 -> 0 0 -1000 -1000
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,  encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0] #B m H
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output) #B m H
        outputs = (layer_output,) + outputs
        return outputs
class PathEncoder(nn.Module):
    def __init__(self,config):
        super(PathEncoder, self).__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lr = nn.Linear(config.hidden_size,config.hidden_size)
        self.elu = nn.ELU()
    def forward(self,hidden,last_hidden):
        hidden = self.lr(hidden)
        hidden = self.layernorm(self.elu(hidden)+last_hidden)
        return hidden
def similar(arr1,arr2):
    farr1 = arr1.ravel()
    farr2 = arr2.ravel()
    numer = torch.sum(farr1*farr2)
    demon = torch.sqrt(torch.sum(farr1 ** 2) * torch.sum(farr2 ** 2))
    sim = numer/demon
    return (sim+1)/2
class TreeHot_B(BertPreTrainedModel):

    def __init__(self, config):
        super(TreeHot_B, self).__init__(config)
        self.bert=BertModel(config=config)
        self.Decoder = DecoderLayer(config)
        self.Decoder_F = DecoderLayer(config)
        self.Decoder_B = DecoderLayer(config)

        self.PathEncoder_F = PathEncoder(config)
        self.PathEncoder_B = PathEncoder(config)
        self.Lr_output = nn.Linear(config.hidden_size,2)

    def multilabel_categorical_crossentropy(self,y_true, y_pred):
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()

    def loss_fun(self,y_true, y_pred):
        """
        y_true:(batch_size, ent_type_size, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size, seq_len, seq_len)
        """
        y_true = torch.nn.functional.one_hot(y_true, 2)
        batch_size, ent_type_size = y_pred.shape[0], y_pred.shape[-1]
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
        loss = self.multilabel_categorical_crossentropy(y_true, y_pred)
        return loss

    def forward(
        self,
        input_ids_question = None,
        attention_mask_question = None,
        input_ids_passage=None,
        attention_mask_passage=None,
        trees=None,
        evidence_dict=None
    ):

        batch_item = int((input_ids_passage.size(0)/input_ids_question.size(0)))
        question = self.bert(
            input_ids_question,
            attention_mask=attention_mask_question,

        ).last_hidden_state
        Hidden_states_0_F = self.PathEncoder_F(question,question)
        Hidden_states_0_B = self.PathEncoder_B(question,question)
        evidence = self.bert(
            input_ids_passage,
            attention_mask=attention_mask_passage,
        ).last_hidden_state

        questions = []
        questions_mask = []
        for q in Hidden_states_0_B:
            questions.extend(batch_item*[q])
        for q in attention_mask_question:
            questions_mask.extend(batch_item*[q])
        questions = torch.stack(questions)
        questions_mask = torch.stack(questions_mask)

        loss,labels,preds = [],[],[]
        V_target = self.Decoder(evidence,questions,questions_mask)[0]
        label = torch.zeros(V_target.size(0))
        for tree in trees:
            for node in tree.leaf_node:
                if node not in tree.nearroot_node:
                    try:
                        label[evidence_dict[node]] = 1
                    except:
                        pass
        labels.extend(label.tolist())
        preds.extend(self.Lr_output(V_target[:,0,:]))

        for i in range(len(trees)):
            tree = trees[i]
            #H0_F = Hidden_states_0_F[i:i+1]
            H0_B = Hidden_states_0_B[i:i+1]
            """
            for k,nodepaths in tree.node2path_F.items():
                for nodepath in nodepaths:
                    hidden = H0_F

                    for path in nodepath[0][1:]:
                        hidden = self.Decoder_F(hidden,evidence[evidence_dict[path]:evidence_dict[path]+1],attention_mask_passage[evidence_dict[path]:evidence_dict[path]+1])[0]
                    preds.extend(self.Lr_output_F(self.Decoder(evidence[batch_item*i:batch_item*(i+1)],hidden,attention_mask_passage[batch_item*i:batch_item*(i+1)])[0][:,0,:]))
                    label = torch.zeros(batch_item)
                    for path in nodepath[1]:
                        label[evidence_dict[path]-batch_item*i] = 1
                    labels.extend(label.tolist())
            """
            stop = False
            for k,nodepaths in tree.node2path_B.items():
                for nodepath in nodepaths:
                    hidden = H0_B
                    for path in nodepath[0]:
                        hidden_ = self.Decoder_B(hidden,evidence[evidence_dict[path]:evidence_dict[path]+1],attention_mask_passage[evidence_dict[path]:evidence_dict[path]+1])[0]
                        hidden = self.PathEncoder_B(hidden_, hidden)
                    preds.extend(self.Lr_output(self.Decoder(evidence[batch_item*i:batch_item*(i+1)],hidden,attention_mask_passage[batch_item*i:batch_item*(i+1)])[0][:,0,:]))
                    label = torch.zeros(batch_item)
                    for path in nodepath[1]:
                        if path != 'Q':
                            label[evidence_dict[path]-batch_item*i] = 1
                    labels.extend(label.tolist())
                    if len(preds)>5000:
                        stop = True
                        break
                if stop:
                    break

        loss = self.loss_fun(torch.tensor(labels,dtype=int,device=questions.device),torch.stack(preds))
        return loss

    def generate_recursion(self,hidden,evidence,attention_mask_passage,node,output,evidence_dict_EN,i,batch_item,ns_list):
        hidden_ = self.Decoder_B(hidden, evidence[node.item():node.item() + 1],
                                attention_mask_passage[node.item():node.item() + 1])[0]
        hidden = self.PathEncoder_B(hidden_, hidden)
        next_nodes = self.Lr_output(self.Decoder(evidence[batch_item * i:batch_item * (i + 1)], hidden,
                                                 attention_mask_passage[batch_item * i:batch_item * (i + 1)])[0][:, 0,
                                    :]).argmax(1)
        for n in ns_list:
            next_nodes[n] = 0
        if len(output)>25:
            return output
        if len(next_nodes.nonzero()) != 0:
            for n in next_nodes.nonzero()[:, 0]:
                output.append([evidence_dict_EN[n.item()+i*batch_item],evidence_dict_EN[node.item()+i*batch_item]])
                output = self.generate_recursion(hidden,evidence,attention_mask_passage,n,output,evidence_dict_EN,i,batch_item,ns_list+[n.item()])
        else:
            output.append(['Q',evidence_dict_EN[node.item()+i*batch_item]])
        return output



    def generate(
        self,
        input_ids_question = None,
        attention_mask_question = None,
        input_ids_passage=None,
        attention_mask_passage=None,
        evidence_dict=None,
        evidence_dict_EN=None,
    ):
        batch_item = int((input_ids_passage.size(0)/input_ids_question.size(0)))
        question = self.bert(
            input_ids_question,
            attention_mask=attention_mask_question,

        ).last_hidden_state
        Hidden_states_0_F = self.PathEncoder_F(question,question)
        Hidden_states_0_B = self.PathEncoder_B(question,question)
        evidence = self.bert(
            input_ids_passage,
            attention_mask=attention_mask_passage,
        ).last_hidden_state
        outputs = []
        questions = []
        questions_mask = []
        for q in Hidden_states_0_B:
            questions.extend(batch_item*[q])
        for q in attention_mask_question:
            questions_mask.extend(batch_item*[q])
        questions = torch.stack(questions)
        questions_mask = torch.stack(questions_mask)

        loss,labels,preds = [],[],[]
        V_target = self.Decoder(evidence,questions,questions_mask)[0]

        nearroot_nodes = self.Lr_output(V_target[:,0,:]).argmax(1)

        for i in range(len(question)):
            #H0_F = Hidden_states_0_F[i:i+1]
            H0_B = Hidden_states_0_B[i:i+1]
            nearroot_nodes_i = nearroot_nodes[i*batch_item:(i+1)*batch_item]
            output = []
            for node in nearroot_nodes_i.nonzero()[:,0]:
                output = self.generate_recursion(H0_B,evidence,attention_mask_passage,node,output,evidence_dict_EN,i,batch_item,[node.item()])
            outputs.append(output)

        return outputs
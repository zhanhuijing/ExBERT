## from https://github.com/graykode/nlp-tutorial/tree/master/5-1.Transformer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import matplotlib.pyplot as plt
import math
import torch.nn.functional as func
from typing import Tuple, Optional
import copy
import os
from torch import Tensor
# def make_batch(sentences):
#     input_batch = [[src_vocab[n] for n in sentences[0].split()]]
#     output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
#     target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
#     return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

## 10
import pickle
def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1)+2, seq.size(1)+2]

    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    # subsequence_mask[:,0:8,0:8] =0
    subsequence_mask[:,0,1] =0

    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


def get_attn_subsequent_mask_img(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1) + 3, seq.size(1) + 3]

    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask[:, 0, 1] = 0
    subsequence_mask[:, 0, 2] = 0
    subsequence_mask[:, 1, 2] = 0

    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]

## 7. ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask,d_k):
        ## 输入进来的维度分别是 [batch_size x n_heads x len_q x d_k]  K： [batch_size x n_heads x len_k x d_k]  V: [batch_size x n_heads x len_k x d_v]
        ##首先经过matmul函数得到的scores形状是 : [batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # print(attn_mask)
        ## 然后关键词地方来了，下面这个就是用到了我们之前重点讲的attn_mask，把被mask的地方置为无限小，softmax之后基本就是0，对q的单词不起作用
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


## 6. MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        ## 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
    def forward(self, Q, K, V, attn_mask):

        ## 这个多头分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value;
        ##输入进来的数据形状： Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        ##下面这个就是先映射，后分头；一定要注意的是q和k分头之后维度是一致额，所以一看这里都是dk
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        ## 输入进行的attn_mask形状是 batch_size x len_q x len_k，然后经过下面这个代码得到 新的attn_mask : [batch_size x n_heads x len_q x len_k]，就是把pad信息重复了n个头上
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)


        ##然后我们计算 ScaledDotProductAttention 这个函数，去7.看一下
        ## 得到的结果有两个：context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q x len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask,self.d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]


## 8. PoswiseFeedForwardNet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)



## 4. get_attn_pad_mask

## 比如说，我现在的句子长度是5，在后面注意力机制的部分，我们在计算出来QK转置除以根号之后，softmax之前，我们得到的形状
## len_input * len*input  代表每个单词对其余包含自己的单词的影响力

## 所以这里我需要有一个同等大小形状的矩阵，告诉我哪个位置是PAD部分，之后在计算计算softmax之前会把这里置为无穷大；

## 一定需要注意的是这里得到的矩阵形状是batch_size x len_q x len_k，我们是对k中的pad符号进行标识，并没有对k中的做标识，因为没必要

## seq_q 和 seq_k 不一定一致，在交互注意力，q来自解码端，k来自编码端，所以告诉模型编码这边pad符号信息就可以，解码端的pad信息在交互注意力层是没有用到的；

def get_attn_pad_mask_img(device, ui_len,seq_q, seq_k,pad_idx):
    seq_q = seq_q.to(device)
    seq_k = seq_k.to(device)
    batch_size ,len_q = seq_q.size()
    batch_size, len_k  = seq_k.size()

    left = torch.zeros(batch_size, ui_len+1).bool().to(device)

    # eq(zero) is PAD token
    pad_attn_mask = torch.cat([left, seq_k.data.eq(pad_idx)],dim=1).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q+ui_len+1, len_k+ui_len+1)  # batch_size x len_q x len_k

def get_attn_pad_mask_img2(device, ui_len,seq_q, seq_k,pad_idx):
    seq_q = seq_q.to(device)
    seq_k = seq_k.to(device)
    batch_size ,len_q = seq_q.size()
    batch_size, len_k  = seq_k.size()

    left = torch.zeros(batch_size, ui_len).bool().to(device)

    # eq(zero) is PAD token
    pad_attn_mask = torch.cat([left, seq_k.data.eq(pad_idx)],dim=1).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q+ui_len+1, len_k+ui_len)  # batch_size x len_q x len_k


def get_attn_pad_mask(device, ui_len,seq_q, seq_k,pad_idx):
    seq_q = seq_q.to(device)
    seq_k = seq_k.to(device)
    batch_size ,len_q = seq_q.size()
    batch_size, len_k  = seq_k.size()

    left = torch.zeros(batch_size, ui_len).bool().to(device)

    # eq(zero) is PAD token
    pad_attn_mask = torch.cat([left, seq_k.data.eq(pad_idx)],dim=1).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q+ui_len, len_k+ui_len)

## 3. PositionalEncoding 代码实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        ## 位置编码的实现其实很简单，直接对照着公式去敲代码就可以，下面这个代码只是其中一种实现方式；
        ## 从理解来讲，需要注意的就是偶数和奇数在公式上有一个共同部分，我们使用log函数把次方拿下来，方便计算；
        ## pos代表的是单词在句子中的索引，这点需要注意；比如max_len是128个，那么索引就是从0，1，2，...,127
        ##假设我的demodel是512，2i那个符号中i从0取到了255，那么2i对应取值就是0,2,4...510
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)## 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)##这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，补长为2，其实代表的就是奇数位置
        ## 上面代码获取之后得到的pe:[max_len*d_model]

        ## 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  ## 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        A = self.pe[:x.size(0), :]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


## 5. EncoderLayer ：包含两个部分，多头注意力机制和前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self,d_model, n_heads, d_k, d_v,d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff

    def forward(self, enc_inputs, enc_self_attn_mask):
        ## 下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x d_model] 需要注意的是最初始的QKV矩阵是等同于这个输入的，去看一下enc_self_attn函数 6.
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


## 2. Encoder 部分包含三个部分：词向量embedding，位置编码部分，注意力层及后续的前馈神经网络

class Encoder(nn.Module):
    def __init__(self, src_emb,d_model,n_layers,n_heads, d_k, d_v,d_ff,dropout, pad_idx, nuser, nitem, args):
        super(Encoder, self).__init__()
        self.src_emb = src_emb
        self.pos_emb = PositionalEncoding(d_model,dropout=dropout) ## 位置编码情况，这里是固定的正余弦函数，也可以使用类似词向量的nn.Embedding获得一个可以更新学习的位置编码
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_k, d_v,d_ff) for _ in range(n_layers)]) ## 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来；
        self.pad_idx = pad_idx
        self.args =args

        self.ui_len =2


    def forward(self, user_emb, item_emb, enc_inputs):
        ## 这里我们的 enc_inputs 形状是： [batch_size x source_len]

        ## 下面这个代码通过src_emb，进行索引定位，enc_outputs输出形状是[batch_size, src_len, d_model]
        device = enc_inputs.device
        # print(device)
        # enc_inputs = enc_inputs.transpose(0, 1)
        enc_outputs = self.src_emb(enc_inputs)

        enc_outputs = torch.cat([user_emb,item_emb, enc_outputs],dim=1)
        #batch_size *长度* dim
        ## 这里就是位置编码，把两者相加放入到了这个函数里面，从这里可以去看一下位置编码函数的实现；3.
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        batch_size = enc_outputs.size(0)
        ##get_attn_pad_mask是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响，去看一下这个函数 4.
        enc_self_attn_mask = get_attn_pad_mask(device,self.ui_len,enc_inputs, enc_inputs,self.pad_idx).to(device)
        enc_self_attns = []
        for layer in self.layers:
            ## 去看EncoderLayer 层函数 5.
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attn = torch.mean(enc_self_attn,dim=1)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns

## 10.
class DecoderLayer(nn.Module):
    def __init__(self,d_model, n_heads, d_k, d_v,d_ff):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.dec_enc_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

## 9. Decoder

class Decoder(nn.Module):
    def __init__(self,tgt_emb, d_model,n_layers, n_heads, d_k, d_v, d_ff, dropout, pad_idx, peter_mask, args):
        super(Decoder, self).__init__()
        self.tgt_emb = tgt_emb
        self.pos_emb = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)])
        self.pad_idx = pad_idx
        self.peter_mask = peter_mask
        self.args =args
        self.ui_len = 2
    def forward(self, user_emb, item_emb, dec_inputs, enc_inputs, enc_outputs,image_feature): # dec_inputs : [batch_size x target_len]

        # dec_inputs = dec_inputs.transpose(0, 1)
        # enc_inputs = enc_inputs.transpose(0, 1)

        dec_outputs = self.tgt_emb(dec_inputs)# [batch_size, tgt_len, d_model]
        if self.args.image_fea:
            dec_outputs = torch.cat([user_emb, item_emb,image_feature,dec_outputs], dim=1)
        else:
            dec_outputs = torch.cat([user_emb,item_emb,dec_outputs],dim=1)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, tgt_len, d_model]
        device = dec_inputs.device
        ## get_attn_pad_mask 自注意力层的时候的pad 部分
        if self.args.image_fea:
            dec_self_attn_pad_mask = get_attn_pad_mask_img(device, self.ui_len,dec_inputs, dec_inputs,pad_idx=self.pad_idx).to(device)
            dec_self_attn_subsequent_mask = get_attn_subsequent_mask_img(dec_inputs).to(device)
            dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
            dec_enc_attn_mask = get_attn_pad_mask_img2(device, self.ui_len, dec_inputs, enc_inputs, self.pad_idx).to(device)
        else:
            dec_self_attn_pad_mask = get_attn_pad_mask(device, self.ui_len,dec_inputs, dec_inputs,pad_idx=self.pad_idx).to(device)
            dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs).to(device)
            dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
            dec_enc_attn_mask = get_attn_pad_mask(device, self.ui_len, dec_inputs, enc_inputs, self.pad_idx).to(device)

        ## get_attn_subsequent_mask 这个做的是自注意层的mask部分，就是当前单词之后看不到，使用一个上三角为1的矩阵

        ## 两个矩阵相加，大于0的为1，不大于0的为0，为1的在之后就会被fill到无限小
        # mask = dec_self_attn_pad_mask + dec_self_attn_subsequent_mask


        ## 这个做的是交互注意力机制中的mask矩阵，enc的输入是k，我去看这个k里面哪些是pad符号，给到后面的模型；注意哦，我q肯定也是有pad符号，但是这里我不在意的，之前说了好多次了哈

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attn = torch.mean(dec_self_attn,dim=1)
            dec_enc_attn = torch.mean(dec_enc_attn, dim=1)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

def generate_peter_mask(src_len, tgt_len):
    total_len = src_len + tgt_len
    mask = generate_square_subsequent_mask(total_len)
    mask[0, 1] = False  # allow to attend for user and item
    return mask


def generate_square_subsequent_mask(total_len):
    mask = torch.tril(torch.ones(total_len, total_len))  # (total_len, total_len), lower triangle -> 1.; others 0.
    mask = mask == 0  # lower -> False; others True
    return mask

class MLP(nn.Module):
    def __init__(self, emsize=512):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(emsize, emsize)
        self.linear2 = nn.Linear(emsize, 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def forward(self, hidden):  # (batch_size, emsize)
        mlp_vector = self.sigmoid(self.linear1(hidden))  # (batch_size, emsize)
        rating = torch.squeeze(self.linear2(mlp_vector))  # (batch_size,)
        return rating



class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = func.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        attns = []

        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attns.append(attn)
        attns = torch.stack(attns)

        if self.norm is not None:
            output = self.norm(output)

        return output, attns




class TransformerDecoderLayer(nn.Module):


    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dec_enc_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = func.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, encoder_out, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, dec_enc_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, dec_self_attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src2, dec_enc_attn = self.dec_enc_attn(src2, encoder_out, encoder_out, attn_mask=None,
                                    key_padding_mask=dec_enc_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, dec_self_attn, dec_enc_attn


class TransformerDecoder(nn.Module):

    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, encoder_out, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, dec_enc_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        dec_self_attns = []
        dec_enc_attns = []

        for mod in self.layers:
            output, dec_self_attn, dec_enc_attn = mod(output, encoder_out, src_mask=mask, src_key_padding_mask=src_key_padding_mask, dec_enc_padding_mask=dec_enc_padding_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        dec_self_attns = torch.stack(dec_self_attns)
        dec_enc_attns = torch.stack(dec_enc_attns)

        if self.norm is not None:
            output = self.norm(output)

        return output, dec_self_attns, dec_enc_attns



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return func.relu
    elif activation == "gelu":
        return func.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))




## 1. 从整体网路结构来看，分为三个部分：编码层，解码层，输出层
class Transformer(nn.Module):
    def __init__(self, args, peter_mask, src_len, tgt_len, src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_k, d_v, d_ff, dropout, pad_idx, nuser, nitem):
        super(Transformer, self).__init__()
        self.word_embeddings = nn.Embedding(src_vocab_size, d_model)  ## 这个其实就是去定义生成一个矩阵，大小是 src_vocab_size * d_model
        # print(src_vocab_size)
        self.user_embeddings = nn.Embedding(nuser, d_model)
        self.item_embeddings = nn.Embedding(nitem, d_model)
        # self.projection = nn.Linear(d_model, tgt_vocab_size) ## 输出层 d_model 是我们解码层每个token输出的维度大小，之后会做一个 tgt_vocab_size 大小的softmax

        self.encoder = Encoder(self.word_embeddings, d_model, n_layers, n_heads, d_k, d_v, d_ff, dropout, pad_idx,nuser, nitem, args)  ## 编码层
        self.decoder = Decoder(self.word_embeddings, d_model, n_layers, n_heads, d_k, d_v, d_ff, dropout, pad_idx, peter_mask, args)  ## 解码层
        # self.decoder_masked = Decoder(self.word_embeddings, d_model, n_layers, n_heads, d_k, d_v, d_ff, dropout, pad_idx, peter_mask, args)  ## 解码层

        #
        # self.pos_encoder_attention = PositionalEncoding(d_model, dropout)  # emsize: word embedding size
        # self.pos_decoder = PositionalEncoding(d_model, dropout)  # emsize: word embedding size
        #
        # attention_layer = TransformerEncoderLayer(d_model, n_heads, args.nhid, dropout)  # nhid: dim_feedforward, one basic layer, including multi-head attention and FFN
        # self.attention_module = TransformerEncoder(attention_layer,args.nlayers)
        #
        # decoder_layers = TransformerDecoderLayer(d_model, n_heads, args.nhid, dropout)  # nhid: dim_feedforward, one basic layer, including multi-head attention and FFN
        # self.transformer_decoder = TransformerDecoder(decoder_layers, args.nlayers)

        self.recommender = MLP(d_model)
        # self.recommender_masked = MLP(d_model)
        self.hidden2token = nn.Linear(d_model, src_vocab_size)
        self.fc = nn.Linear(d_model, d_model)  ## 前馈神经网络-cls
        # self.hidden2token_masked = nn.Linear(d_model, src_vocab_size)
        self.activ = nn.Tanh() ## 激活函数-cls

        self.classifier = nn.Linear(d_model, 2)## cls 这是一个分类层，维度是从d_model到2，对应我们架构图中就是这种：


        self.args = args
        self.image_fea = args.image_fea
        self.ui_len = 2
        self.src_len = src_len
        self.pad_idx = pad_idx
        self.emsize = d_model
        self.weight_path = args.checkpoint
        if peter_mask:
            self.attn_mask = generate_peter_mask(src_len, tgt_len)
        else:
            self.attn_mask = generate_square_subsequent_mask(src_len + tgt_len)
        self.init_weights()



    def init_weights(self):
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        # self.img_embeddings.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.bias.data.zero_()
        #
        # self.hidden2token_masked.weight.data.uniform_(-initrange, initrange)
        # self.hidden2token_masked.bias.data.zero_()

    def get_img(self,dir,names):
        fea= []
        for i in range(len(names)):
            img_name = names[i]
            path_img = os.path.join(dir, img_name)
            mean_img = os.path.join(dir, 'mean')
            if os.path.exists(path_img):
                fea.append(pickle.load(open(path_img, 'rb')))
            else:
                fea.append(pickle.load(open(mean_img, 'rb')))
        fea = torch.stack(fea)
        return fea


    def predict_rating(self, hidden):
        rating = self.recommender(hidden[:, 0, :])  # (batch_size,)
        return rating


    def predict_seq(self, hidden):
        word_prob = self.hidden2token(hidden[:, self.src_len:, :])  # (tgt_len, batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob


    def predict_context(self, hidden):
        context_prob = self.hidden2token(hidden[:, 1, :])  # (batch_size, ntoken)
        log_context_dis = func.log_softmax(context_prob, dim=-1)
        return log_context_dis


    def generate_token(self, hidden):
        word_prob = self.hidden2token(hidden[:, -1, :])  # (batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def predict_side(self, hidden):
        word_prob = self.hidden2token(hidden[:, 2:self.src_len:, :])  # (tgt_len, batch_size, ntoken)
        log_side_prob = func.log_softmax(word_prob, dim=-1)
        return log_side_prob


    def predict_nsp(self,hidden):

        h_pooled = self.activ(self.fc(hidden[:, 3, :])) # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2]

        return logits_clsf

    def predict_rating_masked(self, hidden):
        rating = self.recommender_masked(hidden[:, 0, :])  # (batch_size,)
        return rating

    def predict_seq_masked(self, hidden):
        word_prob = self.hidden2token_masked(hidden[:, self.src_len:, :])  # (tgt_len, batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def predict_context_masked(self, hidden):
        context_prob = self.hidden2token_masked(hidden[:, 1, :])  # (batch_size, ntoken)
        log_context_dis = func.log_softmax(context_prob, dim=-1)
        return log_context_dis


    def generate_token_masked(self, hidden):
        word_prob = self.hidden2token_masked(hidden[:, -1, :])  # (batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def predict_side_masked(self, hidden):
        word_prob = self.hidden2token_masked(hidden[:, 2:self.src_len:, :])  # (tgt_len, batch_size, ntoken)
        log_side_prob = func.log_softmax(word_prob, dim=-1)
        return log_side_prob




    def forward(self, user, item, enc_inputs, dec_inputs, dec_inputs_masked, image_feature, token_flag=False, KD_T = False):
        ## 这里有两个数据进行输入，一个是enc_inputs 形状为[batch_size, src_len]，主要是作为编码段的输入，一个dec_inputs，形状为[batch_size, tgt_len]，主要是作为解码端的输入

        ## enc_inputs作为输入 形状为[batch_size, src_len]，输出由自己的函数内部指定，想要什么指定输出什么，可以是全部tokens的输出，可以是特定每一层的输出；也可以是中间某些参数的输出；
        ## enc_outputs就是主要的输出，enc_self_attns这里没记错的是QK转置相乘之后softmax之后的矩阵值，代表的是每个单词和其他单词相关性；

        user_emb = self.user_embeddings(user).unsqueeze(1)
        item_emb = self.item_embeddings(item).unsqueeze(1)
        enc_outputs, enc_self_attns = self.encoder(user_emb, item_emb, enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(user_emb, item_emb, dec_inputs, enc_inputs, enc_outputs,image_feature)


        rating = self.predict_rating(dec_outputs)
        log_context_dis = self.predict_context(dec_outputs)
        if token_flag:
            log_word_prob = self.generate_token(dec_outputs)  # (tgt_len, batch_size, ntoken)

        else:
            log_word_prob = self.predict_seq(dec_outputs)  # (tgt_len, batch_size, ntoken)
        # log_side_prob = self.predict_side(dec_outputs)  # (tgt_len, batch_size, ntoken)
        logits_clsf = self.predict_nsp(dec_outputs)


        return log_word_prob, rating, log_context_dis, None, dec_outputs, logits_clsf, None, None, None, None, None

# if __name__ == '__main__':
#
#     ## 句子的输入部分，
#     sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
#
#
#     # Transformer Parameters
#     # Padding Should be Zero
#     ## 构建词表
#     src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
#     src_vocab_size = len(src_vocab)
#
#     tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
#     tgt_vocab_size = len(tgt_vocab)
#
#     src_len = 5 # length of source
#     tgt_len = 5 # length of target
#
#     ## 模型参数
#     d_model = 512  # Embedding Size
#     d_ff = 2048  # FeedForward dimension
#     d_k = d_v = 64  # dimension of K(=Q), V
#     n_layers = 6  # number of Encoder of Decoder Layer
#     n_heads = 8  # number of heads in Multi-Head Attention
#
#     model = Transformer()
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     enc_inputs, dec_inputs, target_batch = make_batch(sentences)
#
#     for epoch in range(20):
#         optimizer.zero_grad()
#         outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
#         loss = criterion(outputs, target_batch.contiguous().view(-1))
#         print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
#         loss.backward()
#         optimizer.step()




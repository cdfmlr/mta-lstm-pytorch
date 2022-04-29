#!/usr/bin/env python
# coding: utf-8

# region cli

import argparse

parser = argparse.ArgumentParser(description='MTA-LSTM: a topic-to-essay generator')

parser.add_argument('--data_dir', type=str, default='data/')
parser.add_argument('--model_dir', type=str, default='model_result_multi_layer/')

parser.add_argument('--checkpoint_version_name', type=str, default='')
parser.add_argument('--checkpoint_epoch', type=int, default=0)
parser.add_argument('--checkpoint_type', type=str, default='')

parser.add_argument('--http', action='store_true', help='run HTTP service')
parser.add_argument("--host", type=str, help="server host", default="localhost")
parser.add_argument("--port", type=int, help="listen port", default=8080)

cli_args = parser.parse_args()

# script args
args = {
    'data_dir': cli_args.data_dir or 'data/',
    'model_dir': cli_args.model_dir or 'model_result_multi_layer/',

    'checkpoint_version_name': cli_args.checkpoint_version_name or 'small',
    'checkpoint_epoch': cli_args.checkpoint_epoch or 45,
    'checkpoint_type': cli_args.checkpoint_type or 'trainable',

    'http': cli_args.http,
    'host': cli_args.host,
    'port': cli_args.port,
}


# debug cli
# exit()

# endregion cli


# # MTA-LSTM-PyTorch
#
# This is an implementation of the paper [Topic-to-Essay Generation with Neural Networks]
# (http://ir.hit.edu.cn/~xcfeng/xiaocheng%20Feng's%20Homepage_files/final-topic-essay-generation.pdf). 
# The original work can be found [here](https://github.com/hit-computer/MTA-LSTM), 
# which is implemented in TensorFlow and is totally out-of-date, further more, 
# the owner doesn't seem to maintain it anymore. Therefore, I decided to 
# re-implement it in a simple yet powerful framework, PyTorch.
#
# In this notebook, I'll show you how to build a neural network proposed in the
# paper step by step from scratch.


def log_title(s: str):
    p = 'MTA-LSTM infer: '
    print(p, s)


log_title("start...")

log_title('import packages...')

# ## Import packages
#
# The followings are some packages that'll be used in this work. Make sure you have them installed.

import os
import time
import random
from collections import namedtuple
from typing import List

from gensim.models import KeyedVectors
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import jieba
import jieba.analyse
from aiohttp import web
from aiohttp.web_exceptions import HTTPException

# region cuda-device

log_title('setup device: cuda or cpu')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print('Available cuda:', torch.cuda.device_count())
if torch.cuda.is_available():
    device_num = 0
    deviceName = "cuda:%d" % device_num
    torch.cuda.set_device(device_num)
    print('Current device:', torch.cuda.current_device())
else:
    deviceName = "cpu"

device = torch.device(deviceName)
print('use', device)

# endregion cuda-device

# region dictionary

log_title('build dictionary and pretrained embedding...')

# ## Build a dictionary and pretrained embedding system
#
# Here I'm gonna load the pretrained word2vec vocab and vectors. 
# Please refer to [this notebook]() to he how to train it.
#
# The code `fvec.vectors` is where we get the pretrained vectors.
# `<PAD>`, `<BOS>`, `<EOS>` and `<UNK>` are 4 common tokens
# which stands for *PADding*, *Begin-Of-Sentence*, *End-Of-Sentence* 
# and *UNKnown* respectively. We simply add them into the vocabularies.

save_folder = args['model_dir']

vocab_check_point = os.path.join(save_folder, 'vocab.pkl')
word_vec_check_point = os.path.join(save_folder, 'word_vec.pkl')

if os.path.exists(vocab_check_point) and os.path.exists(word_vec_check_point):
    vocab = torch.load(vocab_check_point)
    word_vec = torch.load(word_vec_check_point)
else:
    file_path = args['data_dir']
    txt_f = os.path.join(file_path, 'composition_mincount_1_305000_vec_original.txt')

    print(f'pretrained files not found: \n'
          f'\t{vocab_check_point}\n'
          f'\t{word_vec_check_point}\n'
          f'\tbuild from {txt_f}')

    fvec = KeyedVectors.load_word2vec_format(txt_f, binary=False)

    word_vec = fvec.vectors

    vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
    vocab.extend(list(fvec.index_to_key))

    word_vec = np.concatenate((np.array([[0] * word_vec.shape[1]] * 4), word_vec))
    word_vec = torch.tensor(word_vec).float()

    del fvec

    torch.save(vocab, vocab_check_point)
    torch.save(word_vec, word_vec_check_point)

print("load dictionary successfully. total %d words" % len(word_vec))

# ## Build a word-index convertor
#
# We don't want to use type of string directly when training, instead we map
# them to a unique index in integer. In text generation phase, we'll then 
# convert them back to string.


word_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_word = {i: ch for i, ch in enumerate(vocab)}


# endregion dictionary

# region model-define

# log_title('define MTA-LSTM model')


# ## Build model: MTA-LSTM
#
# This is the most important part in the notebook.

# ### Bahdanau Attention

class Attention(nn.Module):
    """Implements Bahdanau (MLP) attention"""

    def __init__(self, hidden_size, embed_size):
        super(Attention, self).__init__()

        self.Ua = nn.Linear(embed_size, hidden_size, bias=False)
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.va = nn.Linear(hidden_size, 1, bias=True)
        # to store attention scores
        self.alphas = None

    def forward(self, query, topics, coverage_vector):
        scores = []
        C_t = coverage_vector.clone()
        for i in range(topics.shape[1]):
            proj_key = self.Ua(topics[:, i, :])
            query = self.Wa(query)
            scores += [self.va(torch.tanh(query + proj_key)) * C_t[:, i:i + 1]]

        # stack scores
        scores = torch.stack(scores, dim=1)
        scores = scores.squeeze(2)
        #         print(scores.shape)
        # turn scores to probabilities
        alphas = F.softmax(scores, dim=1)
        self.alphas = alphas

        # mt vector is the weighted sum of the topics
        mt = torch.bmm(alphas.unsqueeze(1), topics)
        mt = mt.squeeze(1)

        # mt shape: [batch x embed], alphas shape: [batch x num_keywords]
        return mt, alphas


# ### Attention Decoder


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, embed_size, num_layers, dropout=0.5):
        super(AttentionDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.dropout = dropout

        # topic attention
        self.attention = Attention(hidden_size, embed_size)

        # lstm
        self.rnn = nn.LSTM(input_size=embed_size * 2,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           dropout=dropout)

    def forward(self, input, output, hidden, phi, topics, coverage_vector):
        # 1. calculate attention weight and mt
        mt, score = self.attention(output.squeeze(0), topics, coverage_vector)
        mt = mt.unsqueeze(1).permute(1, 0, 2)

        # 2. update coverge vector [batch x num_keywords]
        coverage_vector = coverage_vector - score / phi

        # 3. concat input and Tt, and feed into rnn
        output, hidden = self.rnn(torch.cat([input, mt], dim=2), hidden)

        return output, hidden, score, coverage_vector


# ### MTA-LSTM model


LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


class MTALSTM(nn.Module):
    def __init__(self, hidden_dim, embed_dim, num_keywords, num_layers, weight,
                 num_labels, bidirectional, dropout=0.5, **kwargs):
        super(MTALSTM, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.bidirectional = bidirectional
        if num_layers <= 1:
            self.dropout = 0
        else:
            self.dropout = dropout
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.Uf = nn.Linear(embed_dim * num_keywords, num_keywords, bias=False)

        # attention decoder
        self.decoder = AttentionDecoder(hidden_size=hidden_dim,
                                        embed_size=embed_dim,
                                        num_layers=num_layers,
                                        dropout=dropout)

        # adaptive softmax
        self.adaptiveSoftmax = nn.AdaptiveLogSoftmaxWithLoss(
            hidden_dim,
            num_labels,
            cutoffs=[round(num_labels / 20),
                     4 * round(num_labels / 20)])

    def forward(self, inputs, topics, output, hidden=None, mask=None,
                target=None, coverage_vector=None, seq_length=None):
        embeddings = self.embedding(inputs)
        topics_embed = self.embedding(topics)
        ''' calculate phi [batch x num_keywords] '''
        phi = None
        phi = torch.sum(mask, dim=1, keepdim=True) * torch.sigmoid(
            self.Uf(topics_embed.reshape(topics_embed.shape[0], -1).float()))

        # loop through sequence
        inputs = embeddings.permute([1, 0, 2]).unbind(0)
        output_states = []
        attn_weight = []
        for i in range(len(inputs)):
            output, hidden, score, coverage_vector = self.decoder(
                input=inputs[i].unsqueeze(0),
                output=output,
                hidden=hidden,
                phi=phi,
                topics=topics_embed,
                coverage_vector=coverage_vector)  # [seq_len x batch x embed_size]
            output_states += [output]
            attn_weight += [score]

        output_states = torch.stack(output_states)
        attn_weight = torch.stack(attn_weight)

        # calculate loss py adaptiveSoftmax
        outputs = self.adaptiveSoftmax(
            output_states.reshape(-1, output_states.shape[-1]),
            target.t().reshape((-1,)))

        return outputs, output_states, hidden, attn_weight, coverage_vector

    def inference(self, inputs, topics, output, hidden=None, mask=None,
                  coverage_vector=None, seq_length=None):
        embeddings = self.embedding(inputs)
        topics_embed = self.embedding(topics)

        phi = None
        phi = seq_length.float() * torch.sigmoid(
            self.Uf(topics_embed.reshape(topics_embed.shape[0], -1).float()))

        queries = embeddings.permute([1, 0, 2])[-1].unsqueeze(0)

        inputs = queries.permute([1, 0, 2]).unbind(0)
        output_states = []
        attn_weight = []
        for i in range(len(inputs)):
            output, hidden, score, coverage_vector = self.decoder(
                input=inputs[i].unsqueeze(0),
                output=output,
                hidden=hidden,
                phi=phi,
                topics=topics_embed,
                coverage_vector=coverage_vector)  # [seq_len x batch x embed_size]
            output_states += [output]
            attn_weight += [score]

        output_states = torch.stack(output_states)
        attn_weight = torch.stack(attn_weight)

        outputs = self.adaptiveSoftmax.log_prob(
            output_states.reshape(-1, output_states.shape[-1]))
        return outputs, output_states, hidden, attn_weight, coverage_vector

    def init_hidden(self, batch_size):
        # hidden = torch.zeros(num_layers, batch_size, hidden_dim)
        # hidden = LSTMState(torch.zeros(batch_size, hidden_dim).to(device), 
        #                    torch.zeros(batch_size, hidden_dim).to(device))
        hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))
        return hidden

    def init_coverage_vector(self, batch_size, num_keywords):
        # self.coverage_vector = torch.ones([batch_size, num_keywords]).to(device)
        return torch.ones([batch_size, num_keywords]).to(device)
        # print(self.coverage_vector)


# endregion model-define

# region decode-strategy

# ## Greedy decode strategy


def pad_topic(topics):
    topics = [word_to_idx[x] for x in topics]
    topics = torch.tensor(topics)
    print(topics)
    max_num = 5
    size = 1
    ans = np.zeros((size, max_num), dtype=int)
    for i in range(size):
        true_len = min(len(topics), max_num)
        for j in range(true_len):
            print(topics[i])
            ans[i][j] = topics[i][j]
    return ans


def predict_rnn(topics, num_chars, model, idx_to_word, word_to_idx):
    output_idx = [1]
    topics = [word_to_idx[x] for x in topics]
    topics = torch.tensor(topics)
    topics = topics.reshape((1, topics.shape[0]))
    # hidden = torch.zeros(num_layers, 1, hidden_dim)
    # hidden = (torch.zeros(num_layers, 1, hidden_dim).to(device),
    #           torch.zeros(num_layers, 1, hidden_dim).to(device))
    hidden = model.init_hidden(batch_size=1)
    if use_gpu:
        # hidden = hidden.cuda()
        adaptive_softmax.to(device)
        topics = topics.to(device)
    coverage_vector = model.init_coverage_vector(topics.shape[0], topics.shape[1])
    attentions = torch.zeros(num_chars, topics.shape[1])
    for t in range(num_chars):
        X = torch.tensor(output_idx[-1]).reshape((1, 1))
        # X = torch.tensor(output).reshape((1, len(output)))
        if use_gpu:
            X = X.to(device)
        if t == 0:
            output = torch.zeros(1, hidden_dim).to(device)
        else:
            output = output.squeeze(0)
        pred, output, hidden, attn_weight, coverage_vector = model.inference(
            inputs=X, topics=topics, output=output,
            hidden=hidden,
            coverage_vector=coverage_vector,
            seq_length=torch.tensor(50).reshape(1, 1).to(device))
        # print(coverage_vector)
        pred = pred.argmax(dim=1)  # greedy strategy
        attentions[t] = attn_weight[0].data
        # pred = adaptive_softmax.predict(pred)
        if pred[-1] == 2:
            # if pred.argmax(dim=1)[-1] == 2:
            break
        else:
            output_idx.append(int(pred[-1]))
    # output.append(int(pred.argmax(dim=1)[-1]))
    return (
        ''.join([idx_to_word[i] for i in output_idx[1:]]), [idx_to_word[i] for i in output_idx[1:]],
        attentions[:t + 1].t(),
        output_idx[1:])


# ## Beam search strategy

def beam_search(topics, num_chars, model, idx_to_word, word_to_idx, is_sample=False):
    output_idx = [1]
    topics = [word_to_idx[x] for x in topics]
    topics = torch.tensor(topics)
    topics = topics.reshape((1, topics.shape[0]))
    # hidden = torch.zeros(num_layers, 1, hidden_dim)
    # hidden = (torch.zeros(num_layers, 1, hidden_dim).to(device), 
    #           torch.zeros(num_layers, 1, hidden_dim).to(device))
    hidden = model.init_hidden(batch_size=1)
    if use_gpu:
        # hidden = hidden.cuda()
        adaptive_softmax.to(device)
        topics = topics.to(device)
        seq_length = torch.tensor(50).reshape(1, 1).to(device)
    """1"""
    coverage_vector = model.init_coverage_vector(topics.shape[0], topics.shape[1])
    attentions = torch.zeros(num_chars, topics.shape[1])
    X = torch.tensor(output_idx[-1]).reshape((1, 1)).to(device)
    output = torch.zeros(1, hidden_dim).to(device)
    log_prob, output, hidden, attn_weight, coverage_vector = model.inference(
        inputs=X,
        topics=topics,
        output=output,
        hidden=hidden,
        coverage_vector=coverage_vector,
        seq_length=seq_length)
    log_prob = log_prob.cpu().detach().reshape(-1).numpy()
    # print(log_prob[10])
    """2"""
    if is_sample:
        top_indices = np.random.choice(vocab_size, beam_size, replace=False, p=np.exp(log_prob))
    else:
        top_indices = np.argsort(-log_prob)
    """3"""
    beams = [(0.0, [idx_to_word[1]], idx_to_word[1], torch.zeros(1, topics.shape[1]), torch.ones(1, topics.shape[1]))]
    b = beams[0]
    beam_candidates = []
    #     print(attn_weight[0].cpu().data, coverage_vector)
    #     assert False
    for i in range(beam_size):
        word_idx = top_indices[i]
        beam_candidates.append((b[0] + log_prob[word_idx], b[1] + [idx_to_word[word_idx]], word_idx,
                                torch.cat((b[3], attn_weight[0].cpu().data), 0),
                                torch.cat((b[4], coverage_vector.cpu().data), 0), hidden, output.squeeze(0),
                                coverage_vector))
    """4"""
    beam_candidates.sort(key=lambda x: x[0], reverse=True)  # decreasing order
    beams = beam_candidates[:beam_size]  # truncate to get new beams

    for xy in range(num_chars - 1):
        beam_candidates = []
        for b in beams:
            """5"""
            X = torch.tensor(b[2]).reshape((1, 1)).to(device)
            """6"""
            log_prob, output, hidden, attn_weight, coverage_vector = model.inference(
                inputs=X,
                topics=topics,
                output=b[6],
                hidden=b[5],
                coverage_vector=b[7],
                seq_length=seq_length)
            log_prob = log_prob.cpu().detach().reshape(-1).numpy()
            """8"""
            if is_sample:
                top_indices = np.random.choice(vocab_size, beam_size,
                                               replace=False, p=np.exp(log_prob))
            else:
                top_indices = np.argsort(-log_prob)
            """9"""
            for i in range(beam_size):
                word_idx = top_indices[i]
                beam_candidates.append((b[0] + log_prob[word_idx], b[1] + [idx_to_word[word_idx]], word_idx,
                                        torch.cat((b[3], attn_weight[0].cpu().data), 0),
                                        torch.cat((b[4], coverage_vector.cpu().data), 0), hidden, output.squeeze(0),
                                        coverage_vector))
        """10"""
        beam_candidates.sort(key=lambda x: x[0], reverse=True)  # decreasing order
        beams = beam_candidates[:beam_size]  # truncate to get new beams

    """11"""
    if '<EOS>' in beams[0][1]:
        first_eos = beams[0][1].index('<EOS>')
    else:
        first_eos = num_chars - 1
    return (
        ''.join(beams[0][1][:first_eos]),
        beams[0][1][:first_eos],
        beams[0][3][:first_eos].t(),
        beams[0][4][:first_eos])


# endregion decode-strategy

# region load-checkpoint

log_title('config model')

# ## Some configurations

embedding_dim = 100
hidden_dim = 512
lr = 1e-3 * 0.5
momentum = 0.01
clip_value = 0.1
use_gpu = True  # can not be False
num_layers = 2
bidirectional = False
num_keywords = 5
verbose = 1
check_point = 5
beam_size = 2
is_sample = True
vocab_size = len(vocab)
# device = torch.device(deviceName)
loss_function = nn.NLLLoss()
adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
    1000, len(vocab), cutoffs=[round(vocab_size / 20), 4 * round(vocab_size / 20)])

# ## create a model object

model = MTALSTM(hidden_dim=hidden_dim, embed_dim=embedding_dim, num_keywords=num_keywords,
                num_layers=num_layers, num_labels=len(vocab), weight=word_vec, bidirectional=bidirectional)


def params_init_uniform(m):
    if type(m) == nn.Linear:
        y = 0.04
        nn.init.uniform_(m.weight, -y, y)


model.apply(params_init_uniform)

# ## Load previous checkpoint

log_title('Load previous checkpoint')

version_name = args['checkpoint_version_name']
version_epoch_num = args['checkpoint_epoch']
Type = args['checkpoint_type']


def check_point_path(obj_name, save_folder, Type, version_name, epoch_num):
    # f'{save_folder}/{version_name}_{epoch_num}_{Type}_{obj_name}.pt'
    return os.path.join(save_folder, f'{version_name}_{epoch_num}_{Type}_{obj_name}.pt')


model_check_point = check_point_path('model', save_folder, Type, version_name, version_epoch_num)

if os.path.isfile(model_check_point):
    print(f'Loading previous status (ver.{version_name}_{version_epoch_num})...')
    model.load_state_dict(torch.load(model_check_point, map_location='cpu'))
    model = model.to(device)

    print('Load successfully')
else:
    raise RuntimeError(f"load checkpoint failed: model file {model_check_point} doesn't exist.")


# endregion load-checkpoint

# region infer

def infer(input_keywords, method='beam_search', is_sample=False, verbose=False):
    """infer() generates a sentence from input_keywords
    
    example: 
        infer(['现在', '未来', '梦想', '科学', '文化'], method='beam_search', is_sample=True)
    
    :param input_keywords: 
    :param method: "beam_search" or "predict_rnn" (greedy decode strategy)
    :param is_sample: do sample in beam_search
    :param verbose: prints input & output
    :return: str: generated text
    """
    if method == 'beam_search':
        _, output_words, attentions, coverage_vector = beam_search(
            input_keywords, 100, model, idx_to_word, word_to_idx, is_sample=is_sample)
    else:
        _, output_words, attentions, _ = predict_rnn(
            input_keywords, 100, model, idx_to_word, word_to_idx)
    if verbose:
        print('input =', ' '.join(input_keywords))
        print('output =', ' '.join(output_words))
    return ''.join(output_words)


# ## Test some samples

# log_title('Test some samples')
#
# infer(['妈妈', '希望', '长大', '孩子', '母爱'], method='beam_search', is_sample=True, verbose=True)
# infer(['现在', '未来', '梦想', '科学', '文化'], method='beam_search', is_sample=True, verbose=True)
# infer(['春天', '来临', '田野', '聆听', '小路'], method='beam_search', is_sample=True, verbose=True)
# infer(['子女', '父母', '父爱', '无比', '温暖'], method='beam_search', is_sample=True, verbose=True)
# infer(['信念', '人生', '失落', '心灵', '不屈'], method='beam_search', is_sample=True, verbose=True)
# infer(['体会', '母亲', '滴水之恩', '母爱', '养育之恩'], method='beam_search', is_sample=True, verbose=True)


class TextGenerator:
    """['句子', ...] => ['关键词', ...] => '生成文本'
    """

    def _gen_keywords(self, input_sentences: List[str], verbose=False) -> List[str]:
        """['句子', ...] => ['关键词', ...]

        关键词不够 num_keywords（5个）或有的词不在 vocab 中，则随机选择词语来凑。
        """
        if verbose:
            print(f'input_sentences = {input_sentences}')
        temp_text = ' '.join(input_sentences)
        keywords = jieba.analyse.extract_tags(temp_text, withWeight=False, topK=num_keywords)
        # print(f'DEBUG keywords = {keywords}')
        keywords = list(filter(lambda w: w in vocab, keywords))
        # print(f'DEBUG keywords_in_vocab = {keywords}')
        while len(keywords) < num_keywords:
            keywords.append(vocab[random.randint(4, len(vocab))])
        if verbose:
            print(f'keywords = {keywords}')
        return keywords[:num_keywords]

    def _gen_text(self, input_keywords: List[str], verbose=False) -> str:
        """['关键词', ...] => '文本'
        """
        text = infer(input_keywords, method='beam_search', is_sample=True, verbose=verbose)
        text = text.replace('<BOS>', ''). \
            replace('<PAD>', ' '). \
            replace('<EOS>', ''). \
            replace('<UNK>', '!@#¥%')
        return text

    def generate(self, input_sentences: List[str], verbose=False):
        """['句子', ...] => ['关键词', ...] => '生成文本'
        """
        return self._gen_text(
            self._gen_keywords(input_sentences, verbose=verbose), verbose=verbose)

    def __call__(self, input_sentences: List[str], **kwargs):
        return self.generate(input_sentences)


# g = TextGenerator()
#
# print(g([]))
# print(g(['']))
# print(g(['你好']))
# print(g(['你好', '世界']))
# print(g(['你好，世界！']))
# print(g(['你好吗，世界？我可是一点都不好呢！！']))
# print(g(['妈妈', '希望', '长大', '孩子', '母爱']))


# endregion infer

# region server


class HttpServer:
    """基本的 HTTP 服务，基于 aiohttp。

    通过 add_route 方法添加路由。
    调用 run(host, port) 方法开启服务。
    """

    def __init__(self, log=True):
        """"基本的 HTTP 服务

        :param log: True 则使用日志中间件
        """
        self.app = web.Application()
        if log:
            self.app.middlewares.append(self.log_middleware)

    def run(self, host='localhost', port=None):
        web.run_app(self.app, host=host, port=port)

    async def empty_handler(self, request):
        return web.Response()

    def add_route(self, method: str, path: str, handler: callable):
        self.app.add_routes([web.route(method, path, handler)])

    def log(self, level, request: web.Request, response: web.Response, message=''):
        print(f'[{level}] {time.ctime(time.time())} '
              f'{request.method} {request.rel_url} -> {response.status} '
              f'{message}')

    @web.middleware
    async def log_middleware(self, request, handler):
        try:
            response = await handler(request)
            self.log('INFO', request, response)
            return response
        except HTTPException as e:
            self.log('WARN', request, e)
            raise e


class CorsServer(HttpServer):
    """在 HttpServer 的基础上，解决了 CORS 问题。
    """

    def __init__(self, log=True):
        super().__init__(log=log)
        self.app.middlewares.append(self.cors_middleware)

    def add_route(self, method: str, path: str, handler: callable):
        """add_route 时会自动给每个 route 配上一个对应的 options empty_handler
        以解决 cors 问题
        """
        self.app.add_routes([
            web.route(method, path, handler),
            web.route('OPTIONS', path, handler),
        ])

    CORS_HEADERS = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': '*',
        'Access-Control-Allow-Headers': '*',
        'Access-Control-Allow-Credentials': 'true',
    }

    @staticmethod
    @web.middleware
    async def cors_middleware(request, handler):
        """用来解决 cors
        `app = web.Application(middlewares=[cors_middleware])`
        """
        # if request.method == 'OPTIONS':
        #     response = web.Response(text="")
        # else:
        try:
            response = await handler(request)

            for k, v in CorsServer.CORS_HEADERS.items():
                response.headers[k] = v

            return response
        except HTTPException as e:
            for k, v in CorsServer.CORS_HEADERS.items():
                e.headers[k] = v
            raise e


class TextGenServer(CorsServer):
    """主题文本生成的服务
    """

    def __init__(self, log=True, verbose=False):
        super().__init__(log=log)
        self.verbose = verbose
        self.generator = TextGenerator()
        self.add_route('GET', '/gen', self.handle_text_gen)

    async def handle_text_gen(self, request: web.Request):
        seed = request.query.get('s')
        if not seed:
            raise web.HTTPBadRequest(text='require query string s=SEED+SENTENCES+HERE')
        seed = seed.split(' ')
        text = self.generator.generate(seed, verbose=self.verbose)
        return web.Response(text=text)


if __name__ == '__main__' and args['http']:
    server = TextGenServer(verbose=True)
    server.run(host=args['host'], port=args['port'])

# endregion server

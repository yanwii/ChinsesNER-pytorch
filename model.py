# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2018-10-30 15:28:04
'''
import copy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

START_TAG = "START"
STOP_TAG = "STOP"

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):
    def __init__(self, tag_map={"O":0, "B-COM":1, "I-COM":2, "E-COM":3, "START":4, "STOP":5}, vocab_size=20, batch_size=2):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_dim = 128
        self.batch_size = batch_size
        self.embedding_dim = 100
        self.vocab_size = vocab_size
        
        self.tag_size = len(tag_map)
        self.tag_map = tag_map
        
        self.transitions = nn.Parameter(
            torch.randn(self.tag_size, self.tag_size)
        )
        self.word_embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim,
                        num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.tag_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.randn(2, self.batch_size, self.hidden_dim)

    def __get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        length = sentence.shape[1]
        embeddings = self.word_embeddings(sentence).view(self.batch_size, length, self.embedding_dim)

        lstm_out, self.hidden = self.gru(embeddings, self.hidden)
        lstm_out = lstm_out.view(self.batch_size, -1, self.hidden_dim * 2)
        logits = F.softmax(self.hidden2tag(lstm_out), dim=-1)
        # logits = self.hidden2tag(lstm_out)
        return logits
    
    def real_path_score(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([0], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        return score

    def real_path_score_1(self, logits, label):
        '''
        caculate real path score  
        :params logits -> [len_sent * tag_size]
        :params label  -> [1 * len_sent]

        Score = Emission_Score + Transition_Score  
        Emission_Score = logits(0, label[START]) + logits(1, label[1]) + ... + logits(n, label[STOP])  
        Transition_Score = Trans(label[START], label[1]) + Trans(label[1], label[2]) + ... + Trans(label[n-1], label[STOP])  
        '''
        emission_score = sum(map(lambda indic:logits[indic[0], indic[1]], enumerate(label)))
        transition_score = sum(map(lambda index:self.transitions[label[index], label[index+1]], range(len(label)-1)))
        score = emission_score + transition_score
        return score

    def total_score(self, logits, labels):
        forward_var = logits[0]
        for index in range(1, len(logits)):
            alphas_t = []
            logit = logits[index]
            for next_tag in range(self.tag_size):
                emit_score = logit[next_tag].view(1, -1).expand(1, self.tag_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        alpha = log_sum_exp(forward_var)
        return alpha

    def total_score_v1(self, logits, label):
        """
        caculate total score

        :params logits -> [len_sent * tag_size]
        :params label  -> [1 * tag_size]

        SCORE = log(e^S1 + e^S2 + ... + e^SN)
        """
        # label = [0, 1, 2, 2, 3, 0]
        # logits = torch.randn(len(label), self.tag_size)
        self.tag_size = 2
        logits = [
            [0.1, 0.9],
            [0.3, 0.7],
        ]
        transition = [
            [1, 1],
            [1, 1]
        ]
        logits = torch.tensor(logits)
        transition = torch.tensor(transition, dtype=torch.float)

        obs = []
        # [[x0, x1],[x0, x1]] - > [[x0, x0], [x1, x1]]
        previous = logits[0].view(1, -1)
        for index in range(1, len(logits)): 
            previous = previous.expand(self.tag_size, self.tag_size).t()
            obs = logits[index].view(1, -1).expand(self.tag_size, self.tag_size)
            # scores = previous + obs + self.transitions
            scores = previous + obs + transition
            import pdb; pdb.set_trace()
            previous = torch.log(torch.sum(torch.exp(scores), 0))
        # caculate total_scores
        total_scores = torch.log(torch.sum(torch.exp(previous)))
        return total_scores

    def neg_log_likelihood(self, sentence, tags, length):
        logits = self.__get_lstm_features(sentence)
        real_path_score = torch.zeros(1)
        total_score = torch.zeros(1)
        for logit, tag, leng in zip(logits, tags, length):
            logit = logit[:leng]
            tag = tag[:leng]
            real_path_score += self.real_path_score(logit, tag)
            total_score += self.total_score(logit, tag)
        print("real score ", real_path_score)
        print("total score ", total_score)
        print(total_score - real_path_score)
        print("-"*50)
        return total_score - real_path_score

    def forward(self, sentence):
        sentence = torch.tensor(sentence, dtype=torch.long)
        logits = self.__get_lstm_features(sentence)
        for logit in logits:
            score, path = self.__viterbi_decode(logit)
        return score, path
    
    def __viterbi_decode(self, logits):
        backpointers = []
        trellis = torch.zeros(logits.size())
        backpointers = torch.zeros(logits.size(), dtype=torch.long)
        
        trellis[0] = logits[0]
        for t in range(1, len(logits)):
            v = trellis[t - 1].unsqueeze(1).expand_as(self.transitions) + self.transitions
            trellis[t] = logits[t] + torch.max(v, 0)[0]
            backpointers[t] = torch.max(v, 0)[1]
        viterbi = [torch.max(trellis[-1])]
        backpointers = backpointers.numpy()
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        viterbi_score = torch.max(trellis[-1], 0)[0].cpu().tolist()
        return viterbi_score, viterbi

    def __viterbi_decode_v1(self, logits):
        init_prob = 1.0
        trans_prob = self.transitions.t()
        prev_prob = init_prob
        path = []
        for index, logit in enumerate(logits):
            if index == 0:
                obs_prob = logit * prev_prob
                prev_prob = obs_prob
                prev_score, max_path = torch.max(prev_prob, -1)
                path.append(max_path.cpu().tolist())
                continue
            obs_prob = (prev_prob * trans_prob).t() * logit
            max_prob, _ = torch.max(obs_prob, 1)
            _, final_max_index = torch.max(max_prob, -1)
            prev_prob = obs_prob[final_max_index]
            prev_score, max_path = torch.max(prev_prob, -1)
            path.append(max_path.cpu().tolist())
        return prev_score.cpu().tolist(), path


    # https://github.com/napsternxg/pytorch-practice/blob/master/Viterbi%20decoding%20and%20CRF.ipynb
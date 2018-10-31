# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2018-10-31 10:00:03
'''
import torch
import torch.optim as optim

from model import BiLSTM_CRF

START_TAG = "START"
STOP_TAG = "STOP"

ner_model = BiLSTM_CRF(tag_map={"B-ORG": 0, "I-ORG": 1, "O": 2, START_TAG: 3, STOP_TAG: 4, "B-COM":5, "I-COM":6})
optimizer = optim.SGD(ner_model.parameters(), lr=0.01, weight_decay=1e-4)

def train():
    for i in range(300):
        ner_model.zero_grad()

        sentence = torch.tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10], dtype=torch.long)
        tag = torch.tensor([0, 1, 1, 1, 5, 6, 2, 5, 6, 2, 2], dtype=torch.long)

        loss = ner_model.neg_log_likelihood(sentence, tag)
        loss.backward()
        optimizer.step()
        print(loss)

train()
with torch.no_grad():
    sentence = torch.tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10], dtype=torch.long)
    print(ner_model(sentence))
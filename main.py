# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2018-10-31 10:00:03
'''
import torch
import torch.optim as optim

from model import BiLSTMCRF
from data_manager import DataManager


class ChineseNer(object):

    def train(self):

        train_manager = DataManager(batch_size=1)
        ner_model = BiLSTMCRF(
            tag_map=train_manager.tag_map,
            vocab_size=len(train_manager.vocab),
            batch_size=1
        )
        optimizer = optim.Adam(ner_model.parameters())
        # optimizer = optim.SGD(ner_model.parameters(), lr=0.01)
        for epoch in range(10):
            for batch in train_manager.get_batch():
                ner_model.zero_grad()

                sentences, tags, length = zip(*batch)
                sentences = torch.tensor(sentences, dtype=torch.long)
                tags = torch.tensor(tags, dtype=torch.long)
                length = torch.tensor(length, dtype=torch.long)

                loss = ner_model.neg_log_likelihood(sentences, tags, length)

                score, path = ner_model(sentences)
                print("epoch ", epoch)
                print(path)
                print(tags[0].cpu().tolist())
                print(loss)
                print("-"*50)
                loss.backward()
                optimizer.step()

    def predict(self):
        with torch.no_grad():
            ner_model.batch_size = 1
            sentence = torch.tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]], dtype=torch.long)
            print(ner_model(sentence))

cn = ChineseNer()
cn.train()
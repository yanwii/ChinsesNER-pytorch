# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2018-10-31 10:00:03
'''
import torch
import torch.optim as optim

from model import BiLSTM_CRF
from data_manager import DataManager


class ChineseNer(object):

    def train(self):

        train_manager = DataManager(batch_size=1)
        ner_model = BiLSTM_CRF(
            tag_map=train_manager.tag_map,
            vocab_size=len(train_manager.vocab),
            batch_size=1
        )
        optimizer = optim.SGD(ner_model.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(10):
            ner_model.zero_grad()

            for batch in train_manager.get_batch():
                sentences, tags, length = zip(*batch)
                sentences = torch.tensor(sentences, dtype=torch.long)
                tags = torch.tensor(tags, dtype=torch.long)
                length = torch.tensor(length, dtype=torch.long)

                loss = ner_model.neg_log_likelihood(sentences, tags, length)
                loss.backward()
                optimizer.step()
                break
            
            sentences, tags, length = zip(*batch)
            print(ner_model(sentences))
            print(tags)
            exit()
            

    def predict(self):
        with torch.no_grad():
            ner_model.batch_size = 1
            sentence = torch.tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]], dtype=torch.long)
            print(ner_model(sentence))

cn = ChineseNer()
cn.train()
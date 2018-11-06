# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2018-09-28 16:19:07
'''
import copy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

START_TAG = "START"
STOP_TAG = "STOP"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    result = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
    return result

class BiLSTM_CRF(nn.Module):
    
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)

        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        print("total score ", alpha)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
            import pdb; pdb.set_trace()
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        print("real score ", score)
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B-ORG I-ORG I-ORG I-ORG B-COM I-COM O B-COM I-COM O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B-ORG I-ORG O O O O B-ORG I-ORG".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

from data_manager import DataManager
train_manager = DataManager(batch_size=1, data_type="train")

tag_to_ix = copy.deepcopy(train_manager.tag_map)
print(tag_to_ix)
print(train_manager.tag_map)

model = BiLSTM_CRF(len(train_manager.vocab), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer_1 = optim.Adam(model.parameters(), weight_decay=1e-4)


# Check predictions before training
# with torch.no_grad():
#     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
#     precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
#     print(model(precheck_sent))

from model import BiLSTMCRF
ner_model = BiLSTMCRF(
    tag_map=train_manager.tag_map,
    vocab_size=len(train_manager.vocab),
    batch_size=1
)
optimizer_2 = optim.Adam(ner_model.parameters(), weight_decay=1e-4)

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        100):  # again, normally you would NOT do 300 epochs, it is toy data
    for batch in train_manager.get_batch():
        sentences, tags, length = zip(*batch)
        leng = length[0]
        sentence = torch.tensor(sentences[0], dtype=torch.long)[:leng]
        tag = torch.tensor(tags[0], dtype=torch.long)[:leng]

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        ner_model.zero_grad()
        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        # sentence_in = prepare_sequence(sentence, word_to_ix)
        # targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        # sentence_in = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        # targets = torch.tensor([0, 1, 2, 2, 3], dtype=torch.long)
        # Step 3. Run our forward pass.
        loss_1 = model.neg_log_likelihood(sentence, tag)
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss_1.backward()
        optimizer_1.step()
        print(loss_1)
        score, path = model(sentence)
        print(path)
        print(tag.cpu().tolist())
        print("-"*50)
        # sentences = torch.tensor(sentences, dtype=torch.long)
        # tags = torch.tensor(tags, dtype=torch.long)
        # length = torch.tensor(length, dtype=torch.long)
        # loss_2 = ner_model.neg_log_likelihood(sentences, tags, length)
        # score, path = ner_model(sentences)
        # print(path)
        # print(tags[0].cpu().tolist())
        # print(loss_2)
        # print("-"*50)
        # loss_2.backward()
        # optimizer_2.step()


        
# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))

# https://createmomo.github.io/2017/09/23/CRF_Layer_on_the_Top_of_BiLSTM_2/
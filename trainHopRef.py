import sys
import os
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from pre_process import *
from utils import *
from layers import *

# check CPU or GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using " + str(device))


class HopRefQA(torch.nn.Module):
    def __init__(self, hidden_size, batch_size, W_init, config, max_sen_len):
        super(HopRefQA, self).__init__()
        self.embedding = EmbeddingLayer(W_init, config)
        embedding_size = W_init.shape[1] + config['char_filter_size']
        
        self.ga = GatedAttentionLayer() # non-parametrized
        self.gaao = GatedAttentionAttOnly() # non-parametrized
        self.maxAttWords = MaxAttentionWords() # non-parametrized
        self.gating_w = Variable(torch.Tensor([0.5]), requires_grad=True).to(device)
        self.pred = AnswerPredictionLayer() # non-parametrized
        self.hidden_size = hidden_size

        self.context_gru_0 = BiGRU(embedding_size, hidden_size, batch_size)
        self.query_gru_0 = BiGRU(embedding_size, hidden_size, batch_size)

        self.context_gru_1 = BiGRU(2*hidden_size, hidden_size, batch_size)
        self.query_gru_1 = BiGRU(embedding_size, hidden_size, batch_size)

        self.context_gru_2 = BiGRU(2*hidden_size, hidden_size, batch_size)
        self.query_gru_2 = BiGRU(embedding_size, hidden_size, batch_size)

        self.context_gru_3 = BiGRU(2*hidden_size, hidden_size, batch_size)
        self.query_gru_3 = BiGRU(embedding_size, hidden_size, batch_size)

        self.max_sentence = MaxAttSentence(max_sen_len, 2*hidden_size)

    
    def forward(self, context, context_char, query, query_char, candidate, candidate_mask, startends):
        context_embedding, query_embedding = self.embedding(
            context, context_char, 
            query, query_char, 0, 0)

        #-----------------------------------------------------
        #  attention and one hop
        context_lv0 = self.context_gru_0(context_embedding) # B, N, Dh
        query_lv0 = self.query_gru_0(query_embedding) # B, Nq, Dh
        attention = self.gaao(context_lv0, query_lv0) # B, N
        max10 = self.maxAttWords(attention, context_lv0, 10) # B, 15, Dh
        context_lv1 = self.ga(context_lv0, torch.cat([query_lv0, max10], 1)) # cat: B, Nq+15, Dh

        #-----------------------------------------------------
        # attention and second hop
        context_lv2 = self.context_gru_2(context_lv1)
        query_lv2 = self.query_gru_2(query_embedding)
        attention = self.gaao(context_lv2, query_lv2) # B, N
        max10 = self.maxAttWords(attention, context_lv2, 10) # B, 15, Dh
        context_lv3 = self.ga(context_lv2, torch.cat([query_lv2, max10], 1)) # cat: B, Nq+15, Dh

        #-----------------------------------------------------
        context_out = self.context_gru_3(context_lv3)
        query_out = self.query_gru_3(query_embedding)

        candidate_probs = self.pred(
            context_out, 
            query_out, 
            self.hidden_size, 
            candidate, 
            candidate_mask
            )
            
        # output layer
        return candidate_probs # B x Cmax


def main():

    # load config file
    config = load_config(config_path)

    # build dict for token (vocab_dict) and char (vocab_c_dict)
    vocab_dict, vocab_c_dict = build_dict(vocab_path, vocab_char_path)

    # load pre-trained embedding
    # W_init: token index * token embeding
    # embed_dim: embedding dimension
    W_init, embed_dim = load_word2vec_embedding(word_embedding_path, vocab_dict)

    # generate train/valid examples
    train_data, sen_cut_train, max_sen_len_train = generate_examples(train_path, vocab_dict, vocab_c_dict, config, "train")
    dev_data, sen_cut_dev, max_sen_len_dev = generate_examples(valid_path, vocab_dict, vocab_c_dict, config, "dev")
    max_sen_len = max(max_sen_len_train, max_sen_len_dev)
    print("max sentence len: " + str(max_sen_len))

    # construct datasets
    max_word_len = config['max_word_len']
    training_dataset = wikihopDataset(train_data, sen_cut_train, max_word_len)
    training_set = DataLoader(training_dataset, batch_size=config['batch_size'],
                        shuffle=True, num_workers=0, collate_fn=wikihopBatchCollate, drop_last=True)
    dev_dataset = wikihopDataset(dev_data, sen_cut_dev, max_word_len)
    dev_set = DataLoader(dev_dataset, batch_size=config['batch_size'],
                        shuffle=True, num_workers=0, collate_fn=wikihopBatchCollate, drop_last=True)

    #------------------------------------------------------------------------
    # training process begins
    hidden_size = config['nhidden']
    batch_size = config['batch_size']

    model = HopRefQA(hidden_size, batch_size, W_init, config, max_sen_len).to(device)

    if len(sys.argv) > 4 and str(sys.argv[4]) == "load":
        try:
            model.load_state_dict(torch.load(torch_model_p))
            print("saved model loaded")
        except:
            print("no saved model")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    evaluator = ResultEvaluator(config, dev_set, model)

    print("num epoch: " + str(config['num_epochs']))

    for epoch_id in range(config['num_epochs']):
        for batch_id, (batch_train_data, sen_cut_batch) in enumerate(training_set):
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # extract data
            dw, dc, qw, qc, cd, cd_m = extract_data(batch_train_data)

            # forward pass
            cand_probs = model(dw, dc, qw, qc, cd, cd_m, sen_cut_batch) # B x Cmax
            answer = torch.tensor(batch_train_data[10]).type(torch.LongTensor) # B x 1
            loss = criterion(cand_probs, answer)

            # back-prop
            loss.backward()
            optimizer.step()

            # evaluation process
            torch.autograd.no_grad()
            evaluator.step(epoch_id, batch_id, cand_probs, answer)
            torch.autograd.enable_grad()
            
            # save model
            if (epoch_id * batch_size + batch_id) % config['model_save_frequency'] == 0:
                torch.save(model.state_dict(), "model/hopref_{}_{}.pkl".format(epoch_id, batch_id))

            #gc.collect()

            

    print(evaluator.dev_acc_list)


if __name__ == "__main__":
    main()

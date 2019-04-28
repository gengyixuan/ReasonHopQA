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
        self.pred = AnswerPredictionLayer() # non-parametrized
        self.hidden_size = hidden_size

        self.context_gru_0 = BiGRU(embedding_size, hidden_size, batch_size)
        self.query_gru_0 = BiGRU(embedding_size, hidden_size, batch_size)

        self.context_gru_1 = BiGRU(2*hidden_size, hidden_size, batch_size)
        self.query_gru_1 = BiGRU(2*hidden_size, hidden_size, batch_size)

    
    def forward(self, context, context_char, query, query_char, candidate, candidate_mask, startends):
        context_embedding, query_embedding = self.embedding(
            context, context_char, 
            query, query_char, 0, 0)

        #-----------------------------------------------------
        #  attention and one hop
        context = self.context_gru_0(context_embedding) # B, N, Dh
        query = self.query_gru_0(query_embedding) # B, Nq, Dh
        attention = self.gaao(context, query) # B, N
        max20 = self.maxAttWords(attention, context, 20) # B, 15, Dh
        query_0 = torch.cat([query, max20], 1)
        context_0 = self.ga(context, query_0) # cat: B, Nq+15, Dh

        #-----------------------------------------------------
        # attention and second hop
        context = self.context_gru_1(context_0)
        query = self.query_gru_1(query_0)
        attention = self.gaao(context, query) # B, N
        max20 = self.maxAttWords(attention, context, 20) # B, 15, Dh
        query_1 = torch.cat([query, max20], 1)
        context_1 = self.ga(context, query_1) # cat: B, Nq+15, Dh

        #-----------------------------------------------------
        # attention and third hop
        context = self.context_gru_1(context_1)
        query = self.query_gru_1(query_1)
        attention = self.gaao(context, query) # B, N
        max20 = self.maxAttWords(attention, context, 20) # B, 15, Dh
        query_2 = torch.cat([query, max20], 1)
        context_2 = self.ga(context, query_2) # cat: B, Nq+15, Dh

        #-----------------------------------------------------
        # concat final layer representations
        context_out_0 = self.context_gru_1(context_0) # B, N, Dh
        context_out_1 = self.context_gru_1(context_1)
        context_out_2 = self.context_gru_1(context_2)
        query_0 = self.query_gru_1(query_0) # B, Nq, Dh
        query_out_0 = torch.cat([query_0[:,0,:].unsqueeze(1), query_0[:,-1,:].unsqueeze(1)], 1)
        query_1 = self.query_gru_1(query_1) # B, Nq, Dh
        query_out_1 = torch.cat([query_1[:,0,:].unsqueeze(1), query_1[:,-1,:].unsqueeze(1)], 1)
        query_2 = self.query_gru_1(query_2) # B, Nq, Dh
        query_out_2 = torch.cat([query_2[:,0,:].unsqueeze(1), query_2[:,-1,:].unsqueeze(1)], 1)

        context_out = torch.cat([context_out_0, context_out_1, context_out_2], 2) # B,N,3Dh
        query_out = torch.cat([query_out_0, query_out_1, query_out_2], 2) # B,N,3Dh

        # ----------------------------------------------------
        # predict
        candidate_probs = self.pred(
            context_out, 
            query_out, 
            int(self.hidden_size * 3), 
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
                torch.save(model.state_dict(), "model/hopref2_{}_{}.pkl".format(epoch_id, batch_id))

            #gc.collect()

            

    print(evaluator.dev_acc_list)


if __name__ == "__main__":
    main()

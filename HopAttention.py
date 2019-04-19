import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

from modules import EmbeddingLayer
from modules import BiGRU
from modules import GatedAttentionLayer
from modules import GatedAttentionAttOnly
from modules import MaxAttSentence
from modules import HopAttentionLayer
from modules import AnswerPredictionLayer

class HAQA(torch.nn.Module):
    def __init__(self, hidden_size, batch_size, K,  W_init, config):
        super(HAQA, self).__init__()
        self.embedding = EmbeddingLayer(W_init, config)
        embedding_size = W_init.shape[1] + config['char_filter_size']
        
        self.ga = GatedAttentionLayer() # non-parametrized
        self.gaao = GatedAttentionAttOnly() # non-parametrized
        self.ha = HopAttentionLayer() # parametrized
        self.gating_w = Variable(torch.Tensor([0.5]), requires_grad=True)
        self.pred = AnswerPredictionLayer() # non-parametrized
        self.K = K
        self.hidden_size = hidden_size

        self.context_gru_0 = BiGRU(embedding_size, hidden_size, batch_size)
        self.query_gru_0 = BiGRU(embedding_size, hidden_size, batch_size)

        self.context_gru_1 = BiGRU(2*hidden_size, hidden_size, batch_size)
        self.query_gru_1 = BiGRU(embedding_size, hidden_size, batch_size)

        self.context_gru_2 = BiGRU(2*hidden_size, hidden_size, batch_size)
        self.query_gru_2 = BiGRU(embedding_size, hidden_size, batch_size)

        self.context_gru_3 = BiGRU(2*hidden_size, hidden_size, batch_size)
        self.query_gru_3 = BiGRU(embedding_size, hidden_size, batch_size)

        self.max_sentence = MaxAttSentence(100, 128)

    
    def forward(self, context, context_char, query, query_char, candidate, candidate_mask, startends):
        context_embedding, query_embedding = self.embedding(
            context, context_char, 
            query, query_char, 
            0, self.K)

        #-----------------------------------------------------
        # first GA layer
        context_out_0 = self.context_gru_0(context_embedding)
        query_out_0 = self.query_gru_0(query_embedding)
        attention = self.gaao(context_out_0, query_out_0)

        #-----------------------------------------------------
        # max sentence selection
        max_sentence = self.max_sentence(startends, attention, context_out_0) # (batch, max_sentence_len, emb_size)

        # print(max_sentence)
        # return max_sentence
        ha_out = self.ha(max_sentence, context_out_0)
        layer_out_1 = (1 - self.gating_w) * ha_out + self.gating_w * attention

        #-----------------------------------------------------
        context_out_2 = self.context_gru_2(layer_out_1)
        query_out_2 = self.query_gru_2(query_embedding)
        layer_out_2 = self.ga(context_out_2, query_out_2)

        #-----------------------------------------------------
        context_out_3 = self.context_gru_3(layer_out_2)
        query_out_3 = self.query_gru_3(query_embedding)

        candidate_probs = self.pred(
            context_out_3, 
            query_out_3, 
            self.hidden_size, 
            candidate, 
            candidate_mask
            )
            
        # output layer
        return candidate_probs # B x Cmax
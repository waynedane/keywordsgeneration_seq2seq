import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class EmbeddingMatrix(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(EmbeddingMatrix, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        
    def forward(self, unk_inputs): 
        embedded_inputs = self.embedding(unk_inputs).transpose(1,
                                                       0).contiguous()  # [batch_size x seq_len x embedding_dim]->[seq_len x batch_size x embedding_dim]
    
        return embedded_inputs
    
class EncoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, batch_size):
        super(EncoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.gru_t = nn.GRU(embedding_size, hidden_size, bidirectional=True ) #标题GRU
        self.gru_a = nn.GRU(embedding_size, hidden_size, bidirectional=True) #摘要GRU
        self.w_t = nn.Linear(hidden_size*2, hidden_size)
        self.w_a = nn.Linear(hidden_size*2, hidden_size)
        self.v_c = nn.Linear(hidden_size, hidden_size)
        self.batch_size = batch_size
    def forward(self, embedded_inputs, hidden_t, hidden_a):  # 前向传播，两个输入：输入序列和隐层状态
        output_t , hidden_t = self.gru_t(embedded_inputs[:MAX_Length_t], hidden_t) #[length x batch x dense*2]
        output_a , hidden_a = self.gru_a(embedded_inputs[MAX_Length_t:], hidden_a) #[length x batch x dense*2]
        s_0 = self.v_c(self.w_t(output_t[-1])+self.w_a(output_a[-1]))  #[batch x dense]
        return output_t, output_a, output_t[-1].unsqueeze(0), output_a[-1].unsqueeze(0), s_0
    def init_hidden(self):
        hidden_t = Variable(torch.zeros(2, self.batch_size, self.hidden_size)).cuda()  # 初始化隐层参数
        hidden_a = Variable(torch.zeros(2, self.batch_size, self.hidden_size)).cuda()
        return hidden_t, hidden_a
    
class AttnNN(nn.Module):
    def __init__(self, hidden_size):
        super(AttnNN, self).__init__()
        self.hidden_size = hidden_size
        self.Wh_t = nn.Linear(hidden_size*2, hidden_size) # for obtaining e from title encoder hidden states
        self.Ws_t = nn.Linear(hidden_size, hidden_size) # for obtaining e from current state
        self.v_t = nn.Linear(hidden_size, 1)  # for changing to scalar
        self.Wh_a = nn.Linear(hidden_size * 2, hidden_size)  # for obtaining e from abstract encoder hidden states
        self.Ws_a = nn.Linear(hidden_size, hidden_size)  # for obtaining e from current state
        self.v_a = nn.Linear(hidden_size, 1)  # for changing to scalar
        self.wt = nn.Linear(hidden_size*2, hidden_size)
        self.ws = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.gru_v = nn.GRU(hidden_size*2,hidden_size, bidirectional=True)


    def forward(self, d_hidden, outputs_t, outputs_a, hidden_c):
        b = outputs_t.size()[1]  # obtain mini batch
        seq_len_t = len(outputs_t) # obtain  sequence length of title
        seq_len_a = len(outputs_a) # obtain  sequence length of abstract

        attn_energies_t = Variable(torch.zeros(seq_len_t, b)).cuda()
        attn_energies_a = Variable(torch.zeros(seq_len_a, b)).cuda()
        

        for i in range(seq_len_t):
            atten_t_1 = self.Wh_t(outputs_t[i]) + self.Ws_t(
                d_hidden)
            atten_t_2 = self.v_t(F.softplus(atten_t_1))
            attn_energies_t[i] = atten_t_2  #[seq_len x b]
        attn_energies_t = F.softmax(attn_energies_t.transpose(0, 1), dim=1) #[b x seq_len]


        for j in range(seq_len_a):
            atten_a_1 = self.Wh_a(outputs_a[j]) + self.Ws_a(
                d_hidden)
            atten_a_2 = self.v_a(F.softplus(atten_a_1))
            attn_energies_a[j] = atten_a_2
        attn_energies_a = F.softmax(attn_energies_a.transpose(0, 1), dim=1)


        context_t = torch.bmm(attn_energies_t.unsqueeze(1), outputs_t.transpose(1,0))
        context_a = torch.bmm(attn_energies_a.unsqueeze(1), outputs_a.transpose(1,0))
        context_t = context_t.squeeze() #[b x dense*2]
        context_a = context_a.squeeze() #[b x dense*2]
        
    
        attn_energies = Variable(torch.zeros(2, b)).cuda()
        c = torch.cat([context_t.unsqueeze(0),context_a.unsqueeze(0)], 0) #[ 2 x b x dense*2]
        representation_hiddens,_ =  self.gru_v(c,hidden_c)   #[2 x b x dense*2]

        for i in range(2):
            attn_1 = self.wt(representation_hiddens[i]) + self.ws(d_hidden)
            attn_2 = self.v(F.tanh(attn_1))
            attn_energies[i] = attn_2
        attn_energies = F.softmax(attn_energies.transpose(0, 1), dim=1) #[b x2]
        context_vector = torch.bmm(attn_energies.unsqueeze(1),representation_hiddens.transpose(0,1)).squeeze() #[b x 1 x 2] x[b x 2 x dense*2] = [b x 1 x dense*2]
            

        return attn_energies_t, attn_energies_a,  context_vector, attn_energies

    
        

class Attndecoder(nn.Module):
    def __init__(self,  embedding_size, hidden_size, vocab_size, extend_vocab_size, dropout_p=0.1):
        super(Attndecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.extend_vocab_size = extend_vocab_size
        self.dropout_p = dropout_p
        self.embedding_size = embedding_size
        self.gru = nn.GRU(embedding_size, hidden_size, dropout=dropout_p)
      
        self.V1 = nn.Linear(hidden_size * 3, hidden_size * 2)
        self.V2 = nn.Linear(hidden_size * 2, vocab_size)   
        self.wh = nn.Linear(hidden_size*2 , 1)
        self.ws = nn.Linear(hidden_size, 1)
        self.wx = nn.Linear(embedding_size, 1)
        
        self.attn = AttnNN(hidden_size)
    def forward(self, data, target_input ,words_padding_mask, hidden ,outputs_t, outputs_a,hidden_c):
        l_t, b,_= outputs_t.size()  # get length_t &batch size
        l_a = outputs_a.size()[0]
        S_t, hidden = self.gru(target_input, hidden)
        attn_weights_t, attn_weights_a, context,  weights = self.attn(S_t, outputs_t, outputs_a,hidden_c)
        
        weight_1, weight_2 = torch.unbind(weights, dim=1)
        weight_1 = weight_1.repeat(l_t, 1).transpose(0, 1)
        weight_2 = weight_2.repeat(l_a, 1).transpose(0, 1)
        # calculate attention weights distribution
        attn_weights_t = torch.mul(weight_1, attn_weights_t)
        attn_weights_a = torch.mul(weight_2, attn_weights_a)
        attn_dist = torch.cat((attn_weights_t, attn_weights_a),1)
        p_gens = F.sigmoid(self.wh(context) + self.ws(S_t.squeeze()) + self.wx(target_input.squeeze()))
        vocab_dists = F.softmax(self.V2(self.V1(torch.cat([S_t.squeeze(), context], 1))),1)
        vocab_dists = torch.stack([torch.mul(p_gen,dist) for (p_gen, dist) in zip(p_gens,vocab_dists)])
        extra_zeros = Variable(torch.zeros(b, self.extend_vocab_size-self.vocab_size))
        vocab_dists_extended = torch.cat([vocab_dists, extra_zeros], 1)
        renorm_attns = atten_re(attn_dist, Variable(words_padding_mask))
        attn_dists_projected = Variable(torch.stack([torch.sparse.FloatTensor(i.unsqueeze(0), v.data, torch.Size([self.extend_vocab_size])).to_dense() for (i,v) in zip(data,renorm_attns)]))
        attn_dists_projecteds = torch.stack([torch.mul(1-p_gen,attn_dists_projected) for (p_gen, attn_dists_projected) in zip(p_gens,attn_dists_projected)])
        final_output = F.softmax(vocab_dists_extended+attn_dists_projecteds, dim=1)
        return S_t, attn_dist, p_gens, final_output, vocab_dists_extended
    def init_hidden(self):
        hidden_c = Variable(torch.zeros(2,5,self.hidden_size ))
        return hidden_c


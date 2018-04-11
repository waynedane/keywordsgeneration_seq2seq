import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from functions import atten_re


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
    '''
    args:
        embedding_size: dim of input
        hidden_size: dim of encoder's hidden
    '''

    def __init__(self, embedding_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.gru_t = nn.GRU(embedding_size, hidden_size, bidirectional=True)  # 标题GRU
        self.gru_a = nn.GRU(embedding_size, hidden_size, bidirectional=True)  # 摘要GRU
        self.w_t = nn.Linear(hidden_size * 2, hidden_size)
        self.w_a = nn.Linear(hidden_size * 2, hidden_size)
        self.v_c = nn.Linear(hidden_size, hidden_size)

    '''
    input:
        embedded_inputs_ti: title embedding[title_length x batch_size x dense]
        embedded_inputs_te: text embedding[text_length x batch_size x dense]
        hidden_t: init hidden of title encoder
        hiiden_a: init hidden of text encoder

    output:
        output_t: output of title encoder[title_length x batch_size x hidden_size]
        output_a: output of title encoder[text_length x batch_size x hidden_size]
        s_0: init hidden of decoder [batch_size x dense]
    '''

    def forward(self, embedded_inputs_ti, embedded_inputs_te, hidden_t, hidden_a):  # 前向传播，两个输入：输入序列和隐层状态
        output_t, hidden_t = self.gru_t(embedded_inputs_ti, hidden_t)  # [length x batch x dense*2]
        output_a, hidden_a = self.gru_a(embedded_inputs_te, hidden_a)  # [length x batch x dense*2]
        s_0 = self.v_c(self.w_t(output_t[-1]) + self.w_a(output_a[-1]))  # [batch x dense]
        return output_t, output_a, s_0

    def init_hidden(self, embedded_inputs_ti):
        batch_size = embedded_inputs_ti.size()[1]
        hidden_t = Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda()  # 初始化隐层参数
        hidden_a = Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda()
        return hidden_t, hidden_a


class AttnNN(nn.Module):
    def __init__(self, hidden_size):
        super(AttnNN, self).__init__()
        self.hidden_size = hidden_size
        self.Wh_t = nn.Linear(hidden_size * 2, hidden_size)  # for obtaining e from title encoder hidden states
        self.Ws_t = nn.Linear(hidden_size, hidden_size)  # for obtaining e from current state
        self.v_t = nn.Linear(hidden_size, 1)  # for changing to scalar
        self.Wh_a = nn.Linear(hidden_size * 2, hidden_size)  # for obtaining e from abstract encoder hidden states
        self.Ws_a = nn.Linear(hidden_size, hidden_size)  # for obtaining e from current state
        self.v_a = nn.Linear(hidden_size, 1)  # for changing to scalar
        self.wt = nn.Linear(hidden_size * 2, hidden_size)
        self.ws = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.gru_v = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True)

    def forward(self, d_hidden, outputs_t, outputs_a, hidden_c):
        seq_len_t, b, hidden_size_t = outputs_t.size()
        seq_len_a, b_a, hidden_size_a = outputs_a.size()
        #b = outputs_t.size()[1]  # obtain mini batch
        #seq_len_t = len(outputs_t)  # obtain  sequence length of title
        #seq_len_a = len(outputs_a)  # obtain  sequence length of abstract

        #attn_energies_t = Variable(torch.zeros(seq_len_t, b))
        #attn_energies_a = Variable(torch.zeros(seq_len_a, b))

        atten_1 = self.Wh_t(outputs_t.contiguous().view(-1, hidden_size_t)) + self.Ws_t(d_hidden.squeeze()).repeat(seq_len_t, 1)
        atten_1 = self.v_t(F.tanh(atten_1))
        attn_energies_t = F.softmax(atten_1.view(seq_len_t, b).transpose(0, 1), dim=1)

        atten_2 = self.Wh_t(outputs_a.contiguous().view(-1, hidden_size_a)) + self.Ws_a(d_hidden.squeeze()).repeat(seq_len_a, 1)
        atten_2 = self.v_a(F.tanh(atten_2))
        attn_energies_a = F.softmax(atten_2.view(seq_len_a, b).transpose(0, 1), dim=1)

        '''
        for i in range(seq_len_t):
            atten_t_1 = self.Wh_t(outputs_t[i]) + self.Ws_t(
                d_hidden)
            atten_t_2 = self.v_t(F.softplus(atten_t_1))
            attn_energies_t[i] = atten_t_2  # [seq_len x b]
        attn_energies_t = F.softmax(attn_energies_t.transpose(0, 1), dim=1)  # [b x seq_len]

        for j in range(seq_len_a):
            atten_a_1 = self.Wh_a(outputs_a[j]) + self.Ws_a(
                d_hidden)
            atten_a_2 = self.v_a(F.softplus(atten_a_1))
            attn_energies_a[j] = atten_a_2
        attn_energies_a = F.softmax(attn_energies_a.transpose(0, 1), dim=1)
        '''
        context_t = torch.bmm(attn_energies_t.unsqueeze(1), outputs_t.transpose(1, 0))
        context_a = torch.bmm(attn_energies_a.unsqueeze(1), outputs_a.transpose(1, 0))
        context_t = context_t.squeeze()  # [b x dense*2]
        context_a = context_a.squeeze()  # [b x dense*2]

        attn_energies = Variable(torch.zeros(2, b)).cuda()
        c = torch.cat([context_t.unsqueeze(0), context_a.unsqueeze(0)], 0)  # [ 2 x b x dense*2]
        representation_hiddens, _ = self.gru_v(c, hidden_c)  # [2 x b x dense*2]

        for i in range(2):
            attn_1 = self.wt(representation_hiddens[i]) + self.ws(d_hidden)
            attn_2 = self.v(F.tanh(attn_1))
            attn_energies[i] = attn_2
        attn_energies = F.softmax(attn_energies.transpose(0, 1), dim=1)  # [b x2]
        context_vector = torch.bmm(attn_energies.unsqueeze(1), representation_hiddens.transpose(0,
                                                                                                1)).squeeze()  # [b x 1 x 2] x[b x 2 x dense*2] = [b x 1 x dense*2]

        return attn_energies_t, attn_energies_a, context_t,context_vector, attn_energies


class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, extend_vocab_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.extend_vocab_size = extend_vocab_size
        self.dropout_p = dropout_p
        self.embedding_size = embedding_size
        self.gru = nn.GRU(embedding_size, hidden_size, dropout=dropout_p)

        self.V1 = nn.Linear(hidden_size * 5, hidden_size * 2)
        self.V2 = nn.Linear(hidden_size * 2, vocab_size)
        self.wc = nn.Linear(hidden_size * 2, 1)
        self.wh = nn.Linear(hidden_size * 2, 1)
        self.ws = nn.Linear(hidden_size, 1)
        self.wx = nn.Linear(embedding_size, 1)

        self.attn = AttnNN(hidden_size)

    def forward(self, newbatch, target_input, words_padding_mask, d_hidden, outputs_t, outputs_a, hidden_c):
        l_t, b, _ = outputs_t.size()  # get length_t &batch size
        l_a = outputs_a.size()[0]
        S_output,d_hidden= self.gru(target_input, d_hidden)
        attn_weights_t, attn_weights_a, context_t, context, weights = self.attn(d_hidden, outputs_t, outputs_a, hidden_c)

        weight_1, weight_2 = torch.unbind(weights, dim=1)
        weight_1 = weight_1.repeat(l_t, 1).transpose(0, 1)
        weight_2 = weight_2.repeat(l_a, 1).transpose(0, 1)
        # calculate attention weights distribution
        attn_weights_t = torch.mul(weight_1, attn_weights_t)
        attn_weights_a = torch.mul(weight_2, attn_weights_a)
        attn_dist = torch.cat((attn_weights_t, attn_weights_a), 1)
        p_gens = F.sigmoid(self.wh(context) + self.ws(d_hidden.squeeze()) + self.wx(target_input.squeeze()))
        vocab_dists = F.softmax(self.V2(self.V1(torch.cat([d_hidden.squeeze(), context_t, context], 1))), 1)
        vocab_dists = torch.stack([torch.mul(p_gen, dist) for (p_gen, dist) in zip(p_gens, vocab_dists)])
        extra_zeros = Variable(torch.zeros(b, self.extend_vocab_size - self.vocab_size)).cuda()
        vocab_dists_extended = torch.cat([vocab_dists, extra_zeros], 1)
        renorm_attns = atten_re(attn_dist, Variable(words_padding_mask))
        attn_dists_projected = Variable(torch.stack(
            [torch.sparse.FloatTensor(i.unsqueeze(0).cpu(), v.cpu(), torch.Size([self.extend_vocab_size])).to_dense() for (i, v) in
             zip(newbatch, renorm_attns.data)])).cuda()
        attn_dists_projecteds = torch.stack(
            [torch.mul(1 - p_gen, attn_dists_projected) for (p_gen, attn_dists_projected) in
             zip(p_gens, attn_dists_projected)])
        final_output = F.log_softmax(vocab_dists_extended + attn_dists_projecteds, dim=1)
        
        return d_hidden, attn_dist, final_output

    def init_hidden(self, new_batch):
        batch_size = new_batch.size()[0]
        hidden_c = Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda()
        return hidden_c

  class seq2seq(nn.module):
    def __init__(self, embedder, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, title, label, new_batch, words_padding_mask, learning_rate , clip, teacher_forcing_ratio):
        title_embedding = embedder(title)
        label_embedding = embedder(label)
        target_length, batch_size,_ = label_embedding.size()
        hidden_t, hidden_a = encoder.init_hidden()
        output_t, output_a, st = encoder(title_embedding, label_embedding,hidden_t, hidden_a)
        st = st.unsqueeze(0)
        decoder_input = Variable(torch.from_numpy(np.ones([batch, 1],'int64')).cuda())
        decoder_input = embedder(decoder_input)
        use_teacher_forcing= random.random()< teacher_forcing_ratio
        hidden_c = decoder.init_hidden(new_batch)
        outputs = Variable(torch.zeros(target_length, batch_size, vocab_size)).cuda()
        
        if use_teacher_forcing:
            for di in range(target_length):
                st, renorm_attn, final_output = decoder(new_batch,decoder_input, words_padding_mask, st, outputs_t, outputs_a, hidden_c)
                outputs[di] =  final_output 
                decoder_input = embedder(target_unk[di].unsqueeze(1))
        else:
            for di in range(target_length):
                st, renorm_attn, final_output = decoder(new_batch, decoder_input, words_padding_mask,  st, outputs_t, outputs_a, hidden_c)
                outputs[di] =  final_output 
                topv, topi = final_output.data.topk(1)
                decoder_input = embedder(Variable(unk(topi)))
        
        return outputs
        
        
        
        
    
    
    
    
    

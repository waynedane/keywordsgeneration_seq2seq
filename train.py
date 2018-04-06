import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import random
import pickle
import time
import math

from model import EmbeddingMatrix,EncoderRNN, AttnDecoderRNN
from kp20k import KP20K
from functions import seq_mask, atten_re

#Constants
Vocab_Size = 50000
Hidden_Size= 256
Embedding_Size = 256
Batch_Size = 64
teacher_forcing_ratio =0.1
clip = 2.0
learning_rate = 0.0001

with open("dataset/word2index.pkl",'rb') as f:
    dic = pickle.load(f)
List = dic.values()

with open("dataset/word2index_extend.pkl",'rb') as f1:
    oovs = pickle.load(f1)
    Extend_Vocab_Size = len(oovs)+Vocab_Size
    
def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def unk(d):
    size = d.size()
    unk_l = []
    for line in d:
        for i in line:
            if i not in List:
                unk_l.append(3)
            else:
                unk_l.append(i)
    target_unk = torch.LongTensor(unk_l).contigous().view(size)
    return target_unk.cuda()
    
def train(
        title,text , words_padding_mask, target, embedder, encoder, decoder ,embedder_optimzier,
        encoder_optimizer, decoder_optimizer, criterion):
    #Zero gradients of optimizers
    embedder_optimzier.zero_grad()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    loss= 0
    
    input_variable_ti = Variable(unk(title))# input title data for the model
    input_variable_te = Variable(unk(text))
    target = target
    target_unk = unk(target)
    target_variable = Variable(target) # label
    target_unk_variable = Variable(target_unk) #groud truth for teacher forcing 
        
    input_embedded_ti = embedder(input_variable_ti)
    input_embedded_te = embedder(input_variable_te)
    target_length = target.size()[0]
    hidden_t, hidden_a = encoder.init_hidden(title)
    outputs_t, outputs_a, hidden_t, hidden_a, s_t = encoder(input_embedded_ti, input_embedded_te, hidden_t,hidden_a)
    decoder_input = embedder(Variable(torch.from_numpy(np.ones([Batch_Size,1],'int64')).cuda()))
    
    use_teacher_forcing= random.random()< teacher_forcing_ratio
    hidden_c = decoder.init_hidden(target)
    if use_teacher_forcing:
        for di in range(target_length):
            s_t, renorm_attns, final_output = decoder(data, words_padding_mask, decoder_input, s_t, outputs_t, outputs_a,hidden_c)
            loss += criterion(final_output,target_variable[di])
            decoder_input = embedder(Variable(target_unk[di]))
     else:
          for di in range(target_length):
              s_t, renorm_attns, final_output = decoder(data, words_padding_mask, decoder_input, s_t, outputs_t, outputs_a)
              loss += criterion(final_output,target_variable[di])

              topv, topi = final_output.data.topk(1)
              decoder_input = embedder(Variable(unk(topi)))
              loss.backward()
              torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
              torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
              embedder_optimzier.step()
              encoder_optimizer.step()
              decoder_optimizer.step()
      return loss.data[0]/target_length
 
# Initialize models
embedder = EmbeddingMatrix(Vocab_Size, Embedding_Size).cuda()
encoder = EncoderRNN(Embedding_Size, Hidden_Size, Batch_Size).cuda()
decoder = AttnDecoderRNN(Embedding_Size, Batch_Size, Hidden_Size, Vocab_Size, Extend_Vocab_Size,  dropout_p=0.1).cuda()

embedder_optimzier = optim.Adam(embedder.parameters(),lr = learning_rate)
encoder_optimzier = optim.Adam(encoder.parameters(),lr = learning_rate)
decoder_optimzier = optim.Adam(decoder.parameters(),lr = learning_rate)
criterion = nn.NLLLoss

#configring traing

n_epochs = 10000
plot_every = 200
print_every = 1000

plot_losses =[]
print_loss_total = 0
plot_loss_total = 0
#begin
start = time.time()
loss_list=[]
mydataset =  KP20K('dataset',1,True)
train_loader = data.DataLoader(dataset=mydataset,
                                           batch_size=Batch_Size,
                                           shuffle=True,
                                           num_workers=2)

for epoch in range(1,n_epochs+1):

    for batch_index , (data, target) in enumerate(train_loader):
#loading training data

        data = data.cuda()
        title = data[:,:20] #get title
        text = data[:,20:]  #get text
        max_l_ti = torch.nonzero(title).max() # get the max_length of tile of this batch
        max_l_te = torch.nonzero(text).max() # get the max_length of text of this batch
        title = title[:,:max_l_ti+1]
        text = text[:,:max_l_te+1]
        new_batch = torch.cat((title,text),dim=1)
        words_padding_mask = seq_mask(new_batch).cuda()
        target = target.cuda()
        loss = train(
                     title, text, new_batch, target, words_padding_mask,
                     embedder, encoder, decoder ,embedder_optimzier,
                     encoder_optimzier, decoder_optimzier, criterion)

        print_loss_total += loss.data[0]
        plot_loss_total += loss.data[0]
        if epoch == 0: continue
    if epoch%print_every == 0:
        print_loss_avg = print_loss_total / print_every
        loss_list.append(print_loss_avg)
        print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)


    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0
    if epoch %1000 == 0:
        torch.save({'embedder':embedder,'encoder': encoder, 'decoder': decoder}, str(epoch)+'model.pkl')

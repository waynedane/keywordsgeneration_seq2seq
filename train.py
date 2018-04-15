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
from functions import seq_mask
#Constants
Vocab_Size = 50000
Hidden_Size= 256
Embedding_Size = 256
Batch_Size = 32
teacher_forcing_ratio =0.3
clip = 2.0
learning_rate = 0.001

'''
get the value in dict"word2index.pkl"
'''


with open("dataset/word2index.pkl",'rb') as f:
    dic = pickle.load(f)

List = dic.values()

with open("dataset/word2index_extend.pkl",'rb') as f1:
    oovs = pickle.load(f1)
    Extend_Vocab_Size = len(oovs)+Vocab_Size

def unk(d):
    unked = d.clone()
    l,b = unked.size()
    for i in range(l):
       for j in range(b):
           if unked[i][j]>49999:
                unked[i][j]=3
    return unked

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


def train(
        title, text,new_batch, words_padding_mask, target, embedder, encoder, decoder ,embedder_optimzier,
        encoder_optimizer, decoder_optimizer, criterion):
    #Zero gradients of optimizers
    embedder_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    batch = title.size()[0]
    loss = 0
    title_variable = Variable(unk(title))
    text_variable = Variable(unk(text))
    target_unk = Variable(unk(target)) # for teacher forcing and traing
    target_variable = Variable(target) # for computing loss
    #target_unk = Variable(target_unk)
    title_embedded = embedder(title_variable)
    text_embedded = embedder(text_variable)
    target_length = target.size()[0]
    hidden_t, hidden_a = encoder.init_hidden(title_embedded)
    outputs_t, outputs_a, st = encoder(title_embedded,text_embedded, hidden_t, hidden_a)
    st= st.unsqueeze(0)
    
    decoder_input = Variable(torch.from_numpy(np.ones([batch, 1],'int64')).cuda())
    decoder_input = embedder(decoder_input)
    
    
   



    use_teacher_forcing= random.random()< teacher_forcing_ratio
    hidden_c = decoder.init_hidden(new_batch)
    

    if use_teacher_forcing:
        for di in range(target_length):
            st, renorm_attn, final_output = decoder(new_batch,decoder_input, words_padding_mask, st, outputs_t, outputs_a, hidden_c)
            loss += criterion(final_output,target_variable[di])
            decoder_input = embedder(target_unk[di].unsqueeze(1))

    else:
        for di in range(target_length):
            st, renorm_attn, final_output = decoder(new_batch, decoder_input, words_padding_mask,  st, outputs_t, outputs_a, hidden_c)
            loss += criterion(final_output,target_variable[di])

            topv, topi = final_output.data.topk(1)
            decoder_input = embedder(Variable(unk(topi)))



    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    embedder_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]/target_length











# Initialize models
embedder = EmbeddingMatrix(Vocab_Size, Embedding_Size)
encoder = EncoderRNN(Embedding_Size, Hidden_Size)
decoder = AttnDecoderRNN(Embedding_Size, Hidden_Size, Vocab_Size, Extend_Vocab_Size, dropout_p=0.3)

embedder = embedder.cuda()
encoder = encoder.cuda()
decoder = decoder.cuda()

embedder_optimizer = optim.Adam(embedder.parameters(),lr = learning_rate)
encoder_optimizer = optim.Adam(encoder.parameters(),lr = learning_rate, weight_decay=0.0000001)
decoder_optimizer = optim.Adam(decoder.parameters(),lr = learning_rate, weight_decay=0.0000001)
criterion = nn.NLLLoss(ignore_index = 0).cuda()

#configring traing

n_epochs = 40
plot_every = 2
print_every = 5
start = time.time()
plot_losses =[]
print_loss = 0
print_loss_total = 0
plot_loss_total = 0
#begin
loss_list=[]
mydataset =  KP20K('dataset','small',True)
train_loader = data.DataLoader(dataset=mydataset,
                                           batch_size=Batch_Size,
                                           shuffle=True,
                                           num_workers=4)
                                           
print("data loaded!\nstart training!")                                           
for epoch in range(1,n_epochs+1):

    for batch_index , (data, target) in enumerate(train_loader):
#loading training data
        embedder = embedder.train()
        encoder = encoder.train()
        decoder = decoder.train()
        data = data.cuda() #put data to gpu
        title = data[:,:20] #get title gpu
        text = data[:,20:]  #get text gpu
        max_l_ti = torch.nonzero(title).max() # get the max_length of tile of this batch
        max_l_te = torch.nonzero(text).max() # get the max_length of text of this batch
        title = title[:,:max_l_ti+1] #new title batch gpu
        text = text[:,:max_l_te+1] #new text batch gpu
        new_batch = torch.cat((title,text),dim=1) # new batch gpu
        words_padding_mask = seq_mask(new_batch).cuda() #mask
        
     
        target = target.transpose(0,1).cuda()#B*length-> length*B

        loss = train(
                     title, text, new_batch, words_padding_mask,target,
                     embedder, encoder, decoder ,embedder_optimizer,
                     encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss
        print_loss  += loss
    print(print_loss )
    print_loss = 0
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

    if epoch %5 == 0:
        torch.save({'embedder':embedder,'encoder': encoder, 'decoder': decoder}, str(epoch)+'model.pkl')
    if epoch %5 == 0:
        torch.save({'embedder':embedder.state_dict(),'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()},'check/checkpoint.pkl')
     

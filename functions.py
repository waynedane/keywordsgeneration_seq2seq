import torch


def seq_mask(words_input):

    '''
    change all non-zero elements into 1 of a tensor
    args:a tensor(all words to indexes)
    out: a mask list

    example:
    a = torch.LongTensor([[2,1,0,0,0],[1,5,6,0,0]])
    seq_mask(a)
    ...
    1  1  0  0  0
    1  1  1  0  0
    [torch.FloatTensor of size 2x5]
    '''

    size = words_input.size()
    mask = torch.zeros(size)
    for i in torch.nonzero(words_input):
        m=i[0]
        n=i[1]
        mask[m][n]=1
    return mask


def atten_re(attn, mask):

    '''
    renomalize attention distribution
    args: attention distribution,tensor (seq_len xbx1)
          mask tensor (seq_lenx bx1)
    out: renomalize attention distribution,tensor (seq_len xbx1)
    example:
    atten = torch.randn(2,5)
    atten = F.softmax(Variable(atten),dim=1)
    atten
    Variable containing:
 0.4431  0.2424  0.0658  0.1175  0.1312
 0.0910  0.0648  0.0308  0.0608  0.7527
[torch.FloatTensor of size 2x5]
    x= seq_mask(a)
    atten = atten_re(atten,x)
    atten
    Variable containing:
 0.6464  0.3536  0.0000  0.0000  0.0000
 0.4879  0.3472  0.1649  0.0000  0.0000
[torch.FloatTensor of size 2x5]
    '''
    atten = torch.mul(attn,mask)
    length = atten.size()[1]  # get seq_len
    masked_sums = atten.sum(1)
    masked_sums = masked_sums.repeat(length, 1).transpose(1, 0)
    atten = atten / masked_sums
    return atten  # [Batch_Size xSeq_Length ]

import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# print(test_sentence)
trigram = [((test_sentence[i], test_sentence[i + 1]), test_sentence[i + 2])
for i in range(len(test_sentence) - 2)]

vocb = set(test_sentence)
# 建立字典和id的映射关系
word_to_idx = {word:i for i, word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]:word for word in vocb}

class NgramModel(nn.Module):

    def __init__(self, vocb_size, context_size, n_dim):
        super(NgramModel, self).__init__()
        self.n_words = vocb_size
        self.embedding = nn.Embedding(self.n_words, n_dim)
        self.linear1 = nn.Linear(context_size*n_dim, 128)
        self.linear2 = nn.Linear(128, self.n_words)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1, -1)
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)

        log_prob = F.log_softmax(out)

        return log_prob

ngrammodel = NgramModel(len(word_to_idx), CONTEXT_SIZE, 100)
print(ngrammodel)

loss_func = nn.NLLLoss()
optimizer = optim.SGD(ngrammodel.parameters(), lr=1e-3)

for epoch in range(100):
    print('epoch : {}'.format(epoch+1))
    print('*'*20)
    running_loss = 0
    for data in trigram:
        word, label = data
        x = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
        y = Variable(torch.LongTensor([word_to_idx[label]]))

        pre_y = ngrammodel(x)
        loss = loss_func(pre_y, y)
        running_loss += loss.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Loss: {:.6f}'.format(running_loss / len(word_to_idx)))

word, label = trigram[33]
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = ngrammodel(word)
_, predict_label = torch.max(out, 1)
predict_word = idx_to_word[predict_label.data[0]]
print('real word is {}, predict word is {}'.format(label, predict_word))
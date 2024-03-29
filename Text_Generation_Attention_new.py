
import torch
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)
from collections import Counter
from opencc import OpenCC
import jieba
jieba.set_dictionary('dict.txt.big')
class Dataset():
    def __init__(self, path, sequence_length):
        cc = OpenCC('s2t')
        # 讀txt檔案，並且接成一個長字串後，去除空白、去除換行符號後，以字為單位切割
        corpus = cc.convert(open(path, 'r', encoding='UTF-8').read()).replace('\n', '').replace(' ', '').replace('　','')
        self.words = jieba.lcut(corpus)
        print(f'total words: {len(self.words)}')
        self.uniq_words = self.get_uniq_words()
        print(f'uniq words: {len(self.uniq_words)}')
        self.sequence_length = sequence_length
        self.vocab_size = len(self.uniq_words)
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index + self.sequence_length]).to(device),
            torch.tensor([self.words_indexes[index + self.sequence_length]]).to(device),
        )

class Attention(nn.Module):
    """
    A custom self attention layer
    """
    def __init__(self, in_feat,out_feat):
        super().__init__()             
        self.Q = nn.Linear(in_feat,out_feat) # Query
        self.K = nn.Linear(in_feat,out_feat) # Key
        self.V = nn.Linear(in_feat,out_feat) # Value
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        d = K.shape[0] # dimension of key vector
        QK_d = (Q @ K.T)/(d)**0.5
        prob = self.softmax(QK_d)
        attention = prob @ V
        return attention
    
EMBEDDING_DIM = 10
HIDDEN_DIM = 256

class Model(nn.Module):
    def __init__(self,vocab_size,embed_size,seq_size,hidden):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.attention = Attention(embed_size,hidden)
        self.fc1 = nn.Linear(hidden*seq_size,vocab_size) # converting n rows to 1

    def forward(self,x):
        x = self.embed(x)
        x = self.attention(x).view(1,-1)
        x = self.fc1(x)
        #log_probs = F.log_softmax(x,dim=1)
        #return log_probs
        return x
    
from torch import nn, optim
def train(dataset, model, epochs=10):
    model.train()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print('dataset.len:', len(dataset))
    for epoch in range(epochs):
        total_loss = 0
        for i, (x, y) in enumerate(dataset):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print({ 'epoch': epoch, 'loss': total_loss/len(dataset) })

import numpy as np

def predict(dataset, model, text, next_words=100):
    model.eval() 
    words = jieba.lcut(text)
    print('words:', words)
    text = words[:10]
    for i in range(0, next_words):
        x = torch.tensor([dataset.word_to_index[w] for w in text[i:i+100]]).to(device)
        y_pred = model(x)[-1]
        p = torch.softmax(y_pred, dim=0).cpu().detach().numpy()
        word_index = np.random.choice(len(y_pred), p=p)
        text.append(dataset.index_to_word[word_index])
    return ''.join(text)

if __name__ == '__main__':
    sequence_length = 10
    dataset = Dataset('Data/三國演義-smalldata.txt', sequence_length)
    #dataset = Dataset('Data/reddit-cleanjokes.csv', sequence_length)
    #dataloader = DataLoader(dataset, batch_size=1024)
    model = Model(vocab_size = dataset.vocab_size, 
                  embed_size=EMBEDDING_DIM, 
                  seq_size=sequence_length, 
                  hidden=HIDDEN_DIM).to(device)
    train(dataset, model, epochs=1)
    print(predict(dataset, model, text='話說天下大勢，分久必合，合久必分。週末七國分爭，併入於秦。',next_words=2500))
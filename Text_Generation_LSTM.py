#https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
#https://www.kaggle.com/code/purvasingh/text-generation-via-rnn-and-lstms-pytorch

import torch
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from collections import Counter
from opencc import OpenCC
import jieba
jieba.set_dictionary('dict.txt.big')
class Dataset():
    def __init__(self, path, sequence_length):
        cc = OpenCC('s2t')
        # 讀txt檔案，並且接成一個長字串後，去除空白、去除換行符號後，以字為單位切割
        corpus = cc.convert(open(path, 'r', encoding='UTF-8').read()).replace('\n', '').replace(' ', '').replace('　','')
        self.words = jieba.lcut(corpus, cut_all=False)
        print(f'self.words: {self.words[:1000]}')
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
            torch.tensor(self.words_indexes[index+1:index + self.sequence_length+1]).to(device),
        )

class Model(nn.Module):
    def __init__(self, n_vocab):
        super(Model, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 64
        self.num_layers = 3

        self.embedding = nn.Embedding(num_embeddings=n_vocab,embedding_dim=self.embedding_dim)
        bidirectional = True
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.lstm_size, 
                            num_layers=self.num_layers, dropout=0.2, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.lstm_size *= 2
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x):
        embed = self.embedding(x)
        output, state = self.lstm(embed)
        logits = self.fc(output)
        return logits, state

from torch import nn, optim
from torch.utils.data import DataLoader

def train(dataloader, model, epochs=10, sequence_length=4):
    model.train()
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred, _ = model(x)
            loss = loss_function(y_pred.transpose(1, 2), y)
            loss.backward()
            optimizer.step()
            print({ 'epoch': epoch, 'batch': i, 'loss': loss.item() })

import numpy as np

def predict(dataset, model, text, next_words=100):
    model.eval() 
    words = jieba.lcut(text)
    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]]).to(device)
        y_pred, state = model(x)
        last_word_logits = y_pred[0][-1]
        p = torch.softmax(last_word_logits, dim=0).cpu().detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return ''.join(words)

if __name__ == '__main__':
    sequence_length = 3
    dataset = Dataset('Data/三國演義-smalldata.txt', sequence_length)
    #dataset = Dataset('Data/reddit-cleanjokes.csv', sequence_length)
    dataloader = DataLoader(dataset, batch_size=1024)
    model = Model(n_vocab = dataset.vocab_size).to(device)
    train(dataloader, model, epochs=1, sequence_length=sequence_length)
    print(predict(dataset, model, text='兄弟三人',next_words=2500))
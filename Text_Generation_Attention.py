#https://datascience.stackexchange.com/questions/94205/a-simple-attention-based-text-prediction-model-from-scratch-using-pytorch
import torch
import torch.nn as nn
import torch.optim as optim

import random
random.seed(0)
torch.manual_seed(0)

# Sample text for Training
test_sentence = """Thomas Edison. The famed American inventor rose to prominence in the late
19th century because of his successes, yes, but even he felt that these successes
were the result of his many failures. He did not succeed in his work on one of his
most famous inventions, the lightbulb, on his first try nor even on his hundred and
first try. In fact, it took him more than 1,000 attempts to make the first incandescent
bulb but, along the way, he learned quite a deal. As he himself said,
"I did not fail a thousand times but instead succeeded in finding a thousand ways it would not work." 
Thus Edison demonstrated both in thought and action how instructive mistakes can be. 
""".lower().split()

# Build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]

# print the first 3, just so you can see what they look like
print(trigrams[:3])

vocab = list(set(test_sentence))
word_to_ix2 = {word: i for i, word in enumerate(vocab)}

# Number of Epochs
EPOCHS = 25

# SEQ_SIZE is the number of words we are using as a context for the next word we want to predict
SEQ_SIZE = 2

# Embedding dimension is the size of the embedding vector
EMBEDDING_DIM = 10

# Size of the hidden layer
HIDDEN_DIM = 256

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

learning_rate = 0.001
#loss_function = nn.NLLLoss()  # negative log likelihood
loss_function = nn.CrossEntropyLoss()
model = Model(len(vocab),EMBEDDING_DIM,SEQ_SIZE,HIDDEN_DIM)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training
for i in range(EPOCHS):
    total_loss = 0
    for context, target in trigrams:
        # context, target = ['thomas', 'edison.'] the
        
        # step 1: context id generation
        context_idxs = torch.tensor([word_to_ix2[w] for w in context], dtype=torch.long)

        # step 2: setting zero gradient for models
        model.zero_grad()

        # step 3: Forward propogation for calculating log probs
        log_probs = model(context_idxs)

        # step 4: calculating loss
        loss = loss_function(log_probs, torch.tensor([word_to_ix2[target]], dtype=torch.long))

        # step 5: finding the gradients
        loss.backward()

        #step 6: updating the weights
        optimizer.step()

        total_loss += loss.item()
    if i%2==0:
        print("Epoch: ",str(i)," Loss: ",str(total_loss))

# Prediction
with torch.no_grad():
    # Fetching a random context and target 
    rand_val = trigrams[random.randrange(len(trigrams))]
    print(rand_val)
    context = rand_val[0]
    target = rand_val[1]
    
    # Getting context and target index's
    context_idxs = torch.tensor([word_to_ix2[w] for w in context], dtype=torch.long)
    target_idxs = torch.tensor([word_to_ix2[w] for w in [target]], dtype=torch.long)
    print("Acutal indices: ", context_idxs, target_idxs)
    log_preds = model(context_idxs)
    print("Predicted indices: ",torch.argmax(log_preds))
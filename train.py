import numpy as np
import torch
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from neuralnetwork import bagwords, tokenize, stem
from model import nerualnet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['pattern']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))


ignore_words = [',', '?', '/', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bagwords(pattern_sentence, all_words)
    x_train.append(bag)
    label = tag.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)
num_approx = 1000
batchsize = 8
learning_rate = 0.001
inputsize = len(x_train[0])
hiddensize = 8
outputsize = len(tags)
print('training the model.......')


class Chatdataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.xdata = x_train
        self.ydata = y_train

    def __getitem__(self, index):
        return self.xdata[index], self.ydata[index]

    def __len__(self):
        return self.n_samples


dataset = Chatdataset()
trainerloader = DataLoader(dataset=dataset,
                           batch_size=batchsize,
                           shuffle=True,
                           num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nerualnet(inputsize, hiddensize, outputsize).to(device=device)
criteration = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_approx):
    for (words, label) in trainerloader:
        words = words.to(device)
        label = label.to(dtype=torch.long).to(device)
        output = model(words)
        loss = criteration(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch[{epoch+1}/{num_approx}],Loss:{loss.item():.4f}')
print(f'Final loss:{loss.item():.4f}')
data = {
    "model_state": model.state_dict(),
    "input_size": inputsize,
    "hidden_size": hiddensize,
    "output_size": outputsize,
    "all_words": all_words,
    "tags": tags
}
FIlE = "TrainData.pth"
torch.save(data, FIlE)
print(f"Training Complete,File Saved To {FIlE}")
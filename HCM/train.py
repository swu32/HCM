
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset

def train(dataset, model, args):
    model.train()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)# originally 0.001

    for epoch in range(args.max_epochs):
        state_h, state_c = model.init_state(args.sequence_length)

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            # generate prediction of that sequence
            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

        print({ 'epoch': epoch, 'loss': loss.item() })


def predict(dataset, model, next_words=1000, words = [0]):
    predictedword = []
    predictedwordp = []
    model.eval()
    lenword = len(words)

    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[w for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        #word_index = np.argmax(p)
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(word_index)
        predictedwordp.append(p[word_index])
        predictedword.append(word_index)

    return predictedword, predictedwordp


def evaluate_next_word_probability(model, next_word, words = [0]):
    '''Evaluate the probability of the next_word appearing in the sequence '''
    model.eval()
    state_h, state_c = model.init_state(len(words)) # intialize based on the previous words
    x = torch.tensor([[w for w in words[0:]]])
    y_pred, (_,_) = model(x, (state_h, state_c))
    last_word_logits = y_pred[0][-1]
    p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
    return p[next_word]
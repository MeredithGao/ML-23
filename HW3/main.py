from IPython.utils.process import getoutput
import torch
from torch import nn
import utils
from utils import *
import random
import time
import pdb
import json
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

n_letters=58
n_categories=18
n_hidden = 128
n_epochs = 100
print_every = 5000
plot_every = 1000
learning_rate = 0.0005 # If you set this too high, it might explode. If too low, it might not learn

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_shape (int): size of the 1-hot embeddings for each character (this will be 58)
            hidden_layer_width (int): number of nodes in the single hidden layer within the model
            n_classes (int): number of output classes
        """
        super(RNN, self).__init__()
        ### TODO Implement the network architecture
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.i2o = nn.Linear(input_size + hidden_size, output_size, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        """Forward function accepts tensor of input data, returns tensor of output data.
        Modules defined in constructor are used, along with arbitrary operators on tensors
        """
        ### TODO Implement the forward function
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)

        #your function will return the output y(t) and hidden h(t) from equation 1 in the docs
        return output, hidden 

    def initHidden(self):
        """
        This function initializes the first hidden state of the RNN as a zero tensor.
        """
        return torch.zeros(1, self.hidden_size)

def get_xy_pairs(names):
    #TODO 
    #process the names dict and convert into a list of (x,y) pairs. x is a 1-hot tensor of size (num_characters_in_name, 1, n_letters)
    #y is a scalar representing the category of the language, there are 18 languages, assign an index between 0-17 to each language and y represents this index.
    #you may make use of the nameToTensor() function in the utils.py file to help you with this function
    names_list = list(names.keys())
    list_of_pairs = []
    for k, v in names.items():
        for value in names[k]:
            x = utils.nameToTensor(value)
            y = names_list.index(k)
            list_of_pairs.append((x, torch.tensor(y)))

    return list_of_pairs
   

def create_train_and_test_set(list_of_pairs):
    #TODO 
    #process the list of (x,y) pairs and split them 80-20 into train and test set
    #train_x is a list of name embeddings each of size (num_characters_in_name, 1, n_letters), train_y is the correponding list of language category index. Same for test_x and test_y
    X = []
    y = []
    for pair in list_of_pairs:
        X.append(pair[0])
        y.append(pair[1])
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
    return train_x, train_y, test_x, test_y

def stratified_train_and_test_set(list_of_pairs):
    #TODO 
    #process the list of (x,y) pairs and split them 80-20 into train and test set
    #train_x is a list of name embeddings each of size (num_characters_in_name, 1, n_letters), train_y is the correponding list of language category index. Same for test_x and test_y
    X = []
    y = []
    for pair in list_of_pairs:
        X.append(pair[0])
        y.append(pair[1])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    for train_index, test_index in sss.split(X, y):
        train_x, test_x = [X[i] for i in train_index], [X[i] for i in test_index]
        train_y, test_y = [y[i] for i in train_index], [y[i] for i in test_index]
    return train_x, train_y, test_x, test_y

def train(train_x, train_y):
    """train_x and train_y are lists with names and correspoonding labels"""
    loss = 0
    rnn.train()
    for x, y in zip(train_x, train_y):
        hidden = rnn.initHidden()
        for i in range(x.size()[0]):
            output, hidden = rnn(x[i], hidden)

        loss = criterion(torch.log(output + 1e-6), y.unsqueeze(0)) #the unsqueeze converts the scalar y to a 1D tensor
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

def test(test_x, test_y):
    """train_x and train_y are lists with names and correspoonding labels"""
    loss = 0
    losses = []
    with torch.no_grad(): 
        for x, y in zip(test_x, test_y):
            hidden = rnn.initHidden()
            for i in range(x.size()[0]):
                output, hidden = rnn(x[i], hidden)
            
            loss += criterion(torch.log(output + 1e-6), y.unsqueeze(0))

    return loss.item()

# return the rnn model output for a single x
def get_output(rnn, x):
    with torch.no_grad():
        hidden = rnn.initHidden()
        for i in range(x.size()[0]):
            output, hidden = rnn(x[i], hidden)
    return output



# names is your dataset in python dictionary form. Keys are languages and values are list of words belonging to that language
with open('names.json', 'r') as fp:
    names = json.load(fp)

    
# Keep track of losses for plotting
current_loss = 0
all_losses = []
names_list = list(names.keys())

# define model parameters
rnn = RNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

list_of_pairs = get_xy_pairs(names)
train_x, train_y, test_x, test_y = create_train_and_test_set(list_of_pairs)

def cross_entropy_plot():
    all_losses_train = []
    all_losses_test = []
    for epoch in range(1, n_epochs+1):
        loss_train = train(train_x, train_y)
        loss_test = test(test_x, test_y)/len(test_x)
        all_losses_train.append(loss_train)
        #print(f"Epoch={epoch}, Train Loss={loss_train}")
        all_losses_test.append(loss_test)
        #print(f"Epoch={epoch}, Test Loss={loss_test}")

    plt.figure()
    plt.plot(np.linspace(1,n_epochs,n_epochs), all_losses_train, label='Train')
    plt.plot(np.linspace(1,n_epochs,n_epochs), all_losses_test, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy')
    plt.legend()
    plt.show()
    
def plotConfusionMatrix():
        true = []
        pred = []
        with torch.no_grad(): 
            for x, y in zip(test_x, test_y):
                # get true label corresponding to y
                true.append(names_list[y])
                # get the predicted label by chooosing the max from output
                output = get_output(rnn, x)
                topv, topi = torch.max(output, 1)
                pred.append(names_list[topi.item()])

        cm = confusion_matrix(true, pred, normalize='pred')
        cm = np.round(cm, 2)

        cmp = ConfusionMatrixDisplay(cm, display_labels=names.keys())
        fig, ax = plt.subplots(figsize=(10,10))
        cmp.plot(xticks_rotation=90, ax=ax)

        plt.show()

def accuracy(test_x, test_y):
    correct = 0
    with torch.no_grad():
        for x, y in zip(test_x, test_y):
            true_label = names_list[y]
            output = get_output(rnn, x)
            topv, topi = torch.max(output, 1)
            predicted_label = names_list[topi.item()]
            if true_label == predicted_label:
                correct += 1

    return correct / len(test_x)

def stratified_sampling():
    accu_ori = accuracy(test_x, test_y)
    str_train_x, str_train_y, str_test_x, str_test_y = stratified_train_and_test_set(list_of_pairs)
    accu_str = accuracy(str_test_x, str_test_y)

    return accu_ori, accu_str


#saving your model
torch.save(rnn, 'rnn.pt')



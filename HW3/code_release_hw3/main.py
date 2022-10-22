import torch
from utils import *
import random
import time
import pdb
import json
import math

n_letters=58
n_categories=18
n_hidden = 128
n_epochs = 100
print_every = 5000
plot_every = 1000
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_shape (int): size of the 1-hot embeddings for each character (this will be 58)
            hidden_layer_width (int): number of nodes in the single hidden layer within the model
            n_classes (int): number of output classes
        """
        super(RNN).__init__()
        ### TODO Implement the network architecture
        
        
        raise NotImplementedError


    def forward(self, input, hidden):
        """Forward function accepts tensor of input data, returns tensor of output data.
        Modules defined in constructor are used, along with arbitrary operators on tensors
        """
        
        ### TODO Implement the forward function

        
        
        raise NotImplementedError

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

    return list_of_pairs

def create_train_and_test_set(list_of_pairs):
    #TODO 
    #process the list of (x,y) pairs and split them 80-20 into train and test set
    #train_x is a list of name embeddings each of size (num_characters_in_name, 1, n_letters), train_y is the correponding list of language category index. Same for test_x and test_y

    return train_x, train_y, test_x, test_y

rnn = RNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(train_x, train_y):
    """train_x and train_y are lists with names and correspoonding labels"""
    loss = 0
    for x, y in zip(train_x, train_y):
        hidden = rnn.initHidden()
        for i in range(x.size()[0]):
            output, hidden = rnn(x[i], hidden)

        loss += criterion(torch.log(output), y.unsqueeze(0)) #the unsqueeze converts the scalar y to a 1D tensor

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


def test(train_x, train_y):
    """train_x and train_y are lists with names and correspoonding labels"""
    loss = 0
    with torch.no_grad(): 
        for x, y in zip(train_x, train_y):
            hidden = rnn.initHidden()
            for i in range(line_tensor.size()[0]):
                output, hidden = rnn(x[i], hidden)

            loss += criterion(torch.log(output), y.unsqueeze(0)) #the unsqueeze converts the scalar y to a 1D tensor

    return loss


# Keep track of losses for plotting
current_loss = 0
all_losses = []

#names is your dataset in python dictionary form. Keys are languages and values are list of words belonging to that language
with open('names.json', 'r') as fp:
    names = json.load(fp)

list_of_pairs = get_xy_pairs(names)
train_x, train_y, test_x, test_y = create_train_and_test_set(list_of_pairs)

for epoch in range(1, n_epochs + 1):
    train(train_x, train_y)
    test(test_x, test_y)
    

#saving your model
torch.save(rnn, 'rnn.pt')

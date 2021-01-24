import numpy as np 
from utils import *
import random 
import pprint 

data = open('dinos.txt', 'r').read()
data = data.lower()
chars = list(set(data))
chars  = sorted(chars)
data_size, vocab_size = len(data), len(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

#pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(ix_to_char)

def clip(gradients, maxValue):
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    for grad in [dWaa, dWax, dWya, db, dby]:
        np.clip(grad,-maxValue,maxValue,out=grad)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    return gradients

def sample(parameters, char_to_ix, seed):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    x = np.zeros((vocab_size,1))
    a_prev = np.zeros((n_a,1))
    indices = []
    idx = -1
    counter = 0
    newline_character = char_to_ix['\n']

    while(idx!=newline_character and counter!=50):
        a = np.tanh(np.dot(Waa,a_prev) + np.dot(Wax,x) + b)
        y = softmax(np.dot(Wya,a) + by)

        np.random.seed(counter+seed)
        idx  = np.random.choice(range(vocab_size),p=y.ravel())
        indices.append(idx)
        x.fill(0)
        x[idx][0] = 1
        a_prev = a
        seed += 1
        counter += 1

    if counter==50:
        indices.append(newline_character)
    
    return indices


def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    loss, cache = rnn_forward(X,Y,a_prev,parameters)
    gradients, a = rnn_backward(X,Y,parameters,cache)
    gradients = clip(gradients,5)
    parameters = update_parameters(parameters,gradients,learning_rate)

    return loss, gradients, a[len(X)-1]


def model(data, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27, verbose = False):
    n_x, n_y = vocab_size, vocab_size
    parameters = initialize_parameters(n_a, n_x, n_y)
    loss = get_initial_loss(vocab_size, dino_names)

    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    np.random.seed(0)
    np.random.shuffle(examples)

    a_prev = np.zeros((n_a,1))

    for j in range(num_iterations):
        idx = j%len(examples)
        single_example = list(examples[idx])
        single_example = [char_to_ix[ch] for ch in single_example]
        X = [None] + single_example

        ix_newline = char_to_ix['\n']
        Y = single_example + [ix_newline]

        curr_loss, gradients, a_prev = optimize(X,Y,a_prev,parameters,0.01)
        loss = smooth(loss,curr_loss)
    
    return parameters


parameters = model(data, ix_to_char, char_to_ix, verbose=True)

def dino_print(seed, num_of_dino):
    for i in range(num_of_dino):
        idx = sample(parameters,char_to_ix,seed)
        print_sample(idx,ix_to_char)
        seed += 1


if __name__ == '__main__':
    seed = int(input("Enter any random natural number "))
    for i in range(10):
        idx = sample(parameters,char_to_ix,seed)
        print_sample(idx,ix_to_char)
        seed += 1


import sys, os
import pickle
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = \
            load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("./code_given/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y  = softmax(a3)
    return y

def forward(network, x_set, t_set, acc_cnt):
    for (x,t) in zip(x_set, t_set):
        y = predict(network, x)
        p = np.argmax(y)
        if p == t:
            acc_cnt += 1
    return acc_cnt

def forward_batch(network, x_set, t_set, acc_cnt, bat_size):
    for i in range(0, len(x_set), bat_size):
        x_batch = x_set[i:i+bat_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        acc_cnt += np.sum(p == t_set[i:i+bat_size])
    return acc_cnt

if __name__ == '__main__':
    sys.path.append(os.path.curdir + '/code_given')
    from dataset.mnist import load_mnist

    accuracy_cnt = 0
    batch_size = 100
    x_set, t_set = get_data()
    network = init_network()
    #accuracy_cnt = forward(network, x_set, t_set, accuracy_cnt)
    accuracy_cnt = forward_batch(network, x_set, t_set, accuracy_cnt, batch_size)

    print("Accuracy:" + str(float(accuracy_cnt) / len(x_set)))

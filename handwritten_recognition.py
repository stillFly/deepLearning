import sys, os
import numpy as np
import matplotlib.pylab as plt

if __name__ == '__main__':
    sys.path.append(os.path.curdir + '\\code_given')
    from dataset.mnist import load_mnist
    #print(sys.path)
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True)

    img = x_train[0]
    label = t_train[0]
    print(label)

    img = img.reshape(28, 28)
    plt.imshow(img)
    plt.show()

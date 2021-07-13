import numpy as np
import time
import os

#np.float128 = np.float


def Sigmoid(Z):
    return 1 / (1 + np.exp(-Z, dtype=np.float128))


def dSigmoid(Z):
    s = 1 / (1 + np.exp(-Z, dtype=np.float128))
    dZ = s * (1 - s)
    return dZ


def Relu(Z):
    return np.maximum(0, Z, dtype=np.float128)


def dRelu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


class dlnet:
    def __init__(self, x, y):
        self.X = x
        self.Y = y
        self.Yh = np.zeros((1, self.Y.shape[1]))
        self.L = 2
        self.dims = [3, 7, 1]
        self.param = {}
        self.ch = {}
        self.grad = {}
        self.loss = []
        self.lr = 0.003
        self.sam = self.Y.shape[1]

    def nInit(self):
        np.random.seed(1)
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0])
        self.param['b1'] = np.zeros((self.dims[1], 1))
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1])
        self.param['b2'] = np.zeros((self.dims[2], 1))
        return

    def forward(self):
        Z1 = self.param['W1'].dot(self.X) + self.param['b1']
        A1 = Relu(Z1)
        self.ch['Z1'], self.ch['A1'] = Z1, A1

        Z2 = self.param['W2'].dot(A1) + self.param['b2']
        A2 = Sigmoid(Z2)
        self.ch['Z2'], self.ch['A2'] = Z2, A2
        self.Yh = A2
        loss = self.nloss(A2)
        return self.Yh, loss

    def nloss(self, Yh):
        loss = (1. / self.sam) * (-np.dot(self.Y, np.log(Yh).T) - np.dot(1 - self.Y, np.log(1 - Yh).T))
        return loss

    def backward(self):
        dLoss_Yh = - (np.divide(self.Y, self.Yh, dtype=np.float128) - np.divide(1 - self.Y, 1 - self.Yh,
                                                                                dtype=np.float128))

        dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z2'])
        dLoss_A1 = np.dot(self.param["W2"].T, dLoss_Z2)
        dLoss_W2 = 1. / self.ch['A1'].shape[1] * np.dot(dLoss_Z2, self.ch['A1'].T)
        dLoss_b2 = 1. / self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1], 1]))

        dLoss_Z1 = dLoss_A1 * dRelu(self.ch['Z1'])
        dLoss_A0 = np.dot(self.param["W1"].T, dLoss_Z1)
        dLoss_W1 = 1. / self.X.shape[1] * np.dot(dLoss_Z1, self.X.T)
        dLoss_b1 = 1. / self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1], 1]))

        self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
        self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2

    def gd(self, X, Y, iter=3000):
        np.random.seed(1)

        self.nInit()

        for i in range(0, iter):
            Yh, loss = self.forward()
            self.backward()

            if i % 10_000 == 0:
                print("Cost after iteration %i: %f" % (i, loss))
                self.loss.append(loss)

        return

    def pred(self, x, y):
        self.X = x
        self.Y = y
        comp = np.zeros((1, x.shape[1]))
        pred, loss = self.forward()

        for i in range(0, pred.shape[1]):
            if pred[0, i] > 0.5:
                comp[0, i] = 1
            else:
                comp[0, i] = 0

        out = {0.: {True: 0, False: 0}, 1.: {True: 0, False: 0}}

        for i in range(0, pred.shape[1]):
            out[y[0][i]][y[0][i] == comp[0][i]] += 1

        print(out)

        TP = out[1.][True]
        TN = out[0.][True]
        FP = out[1.][False]
        FN = out[0.][False]

        # print("Acc: " + str(np.sum((comp == y) / x.shape[1])))
        acc = (TP + TN) / (TP + TN + FP + FN)
        print(f'acc = {acc:.2f}')

        spec = TN / (TN + FP)
        print(f'spec = {spec:.2f}')

        sens = TP / (TP + FN)
        print(f'sens = {sens:.2f}')

        return comp

    def save(self):
        dir = 'out-{}'.format(time.time())
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'out', dir)
        os.mkdir(path)

        np.savetxt(os.path.join(path,'W1.csv'), self.param['W1'], delimiter=',')
        np.savetxt(os.path.join(path,'b1.csv'), self.param['b1'], delimiter=',')
        np.savetxt(os.path.join(path,'W2.csv'), self.param['W2'], delimiter=',')
        np.savetxt(os.path.join(path,'b2.csv'), self.param['b2'], delimiter=',')


if __name__ == '__main__':
    all_data = np.loadtxt('in.csv', delimiter=',', encoding='utf-8-sig', dtype=np.float128)
    num_rows, num_cols = all_data.shape
    x = (all_data[:, 0:num_cols - 1]).transpose()
    y = all_data[:, num_cols - 1].reshape(1, num_rows)

    nn = dlnet(x, y)
    nn.gd(x, y, iter=100_000)
    nn.save()

    # pred_train = nn.pred(x, y)
    pred_test = nn.pred(x, y)

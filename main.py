import numpy as np
import time
import os
import matplotlib.pyplot as plt
import math

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

        return Yh

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

        tpr, fpr = self.true_false_positive(comp, y)

        print(f'tpr={tpr}, fpt={fpr}')

        out = {0.: {True: 0, False: 0}, 1.: {True: 0, False: 0}}

        for i in range(0, pred.shape[1]):
            out[y[0][i]][y[0][i] == comp[0][i]] += 1

        print(out)

        TP = out[1.][True]
        TN = out[0.][True]
        FP = out[1.][False]
        FN = out[0.][False]

        err_p = (FP + FN)/pred.shape[1]
        print(f'err % = {err_p:.2f}')

        interval = 1.96 * math.sqrt((err_p * (1 - err_p)) / pred.shape[1])
        print(f'%95 CI = {interval:.3f}')

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

    def nLoad(self, loadCsv):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'out', loadCsv)

        self.param['W1'] = np.loadtxt(os.path.join(path,'W1.csv'), delimiter=',').reshape(self.dims[1], self.dims[0])
        self.param['b1'] = np.loadtxt(os.path.join(path,'b1.csv'), delimiter=',').reshape(self.dims[1], 1)
        self.param['W2'] = np.loadtxt(os.path.join(path,'W2.csv'), delimiter=',').reshape(self.dims[2], self.dims[1])
        self.param['b2'] = np.loadtxt(os.path.join(path,'b2.csv'), delimiter=',').reshape(self.dims[2], 1)

    def true_false_positive(self, threshold_vector, y_test):
        true_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 1)
        true_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 0)
        false_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 0)
        false_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 1)

        tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
        fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

        return tpr, fpr

    def roc(self, probabilities, y_test, partitions=100):
        roc = np.array([])
        for i in range(partitions + 1):
            threshold_vector = np.greater_equal(probabilities, i / partitions).astype(int)
            tpr, fpr = self.true_false_positive(threshold_vector, y_test)
            roc = np.append(roc, [fpr, tpr])

        return roc.reshape(-1, 2)



if __name__ == '__main__':
    all_data = np.loadtxt('in.csv', delimiter=',', encoding='utf-8-sig', dtype=np.float128)
    num_rows, num_cols = all_data.shape
    x = (all_data[:, 0:num_cols - 1]).transpose()
    y = all_data[:, num_cols - 1].reshape(1, num_rows)

    death = np.count_nonzero(y[0])
    alive = num_rows - death
    dDeath = (death * 100)/num_rows
    dAlive = (alive * 100)/num_rows

    print(f'Distribution live={dAlive:.2f}%({alive}), die={dDeath:.2f}%({death})')

    ### generate
    # nn = dlnet(x, y)
    # prob = nn.gd(x, y, iter=10_000)
    # nn.save()

    ### LOAD
    nn = dlnet(x, y)
    nn.nLoad('out-1626170557.7618911')
    prob, loss = nn.forward()



    ## prob setup plot

    partitions = 1000
    ROC = nn.roc(prob, y, partitions=partitions)
    fpr, tpr = ROC[:, 0], ROC[:, 1]

    ### PLOT
    plt.figure(figsize=(15, 7))
    plt.scatter(ROC[:, 0], ROC[:, 1], color='#0F9D58', s=100)
    plt.title('ROC Curve', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

    rectangle_roc = 0
    for k in range(partitions):
        rectangle_roc = rectangle_roc + (fpr[k] - fpr[k + 1]) * tpr[k]
        # plt.plot([fpr[j], fpr[j]], [ tpr[j], 0], 'k-', lw=2, color='#4285F4')

    print(f'AOC={rectangle_roc}')

    # pred_train = nn.pred(x, y)
    pred_test = nn.pred(x, y)
    # plt.show()

import numpy as np

class mlp(object):
    def __init__(self, reg_lambda, epsilon, num_layer, X_train, y):
        self.epsilon = epsilon
        self.reg_lambda = reg_lambda
        self.num_layer = num_layer
        self.num_train = X_train.shape[0]
        self.train_dim = X_train.shape[1]
        self.predict_dim = y.shape[0]
        # pass

    def getLoss(self, X, y):
        # pass
        self.probs = np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
        self._loss = np.sum(-np.log(self.probs[range(X.shape[0]), list(y)])) / X.shape[0]
        return self._loss

    def predict(self):
        pass

    def forword(self,X, layer):
        # pass
        res = X.dot(layer['W']) + layer['b']
        return np.tanh(res)

    def backForword(self, X):
        # pass
        grad = self.probs[range(X.shape[0]), list(y)] -= 1

    def layer_init(self, nn_input_dim, nn_hdim):
        # pass
        # nn_input_dim 输入参数维度
        # nn_hdim 输出参数维度
        # 用随机值初始化参数。我们需要学习这些参数
        np.random.seed(0)
        W = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
        b = np.zeros((1, nn_hdim))
        layer = {}
        layer = {'W':W, 'b':b}
        return layer

    def updateLayer(self, layer, dW):
        # pass
        layer['W'] += dW
        return layer

    def get_grad(self):
        pass

    def buildModel(self):
        # how many layers
        # init layers
        # get loss
        # get dw
        # update layer's w
        pass
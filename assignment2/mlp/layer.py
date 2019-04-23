import numpy as np

class layer(object):
    
    def __init__(self, input_dim, output_dim):
        # 用随机值初始化参数。我们需要学习这些参数
        np.random.seed(0)
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.zeros((1, output_dim))

        def forward(self, X):
            res = X.dot(self.W) + self.b
            return np.tanh(res)

        def backForward(self,probs, X, y):
            delta = probs

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

newsgroups_train = fetch_20newsgroups(subset='train',  categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',  categories=categories)

# pprint(newsgroups_train.data[0])

num_train = len(newsgroups_train.data)
num_test  = len(newsgroups_test.data)

vectorizer = TfidfVectorizer(max_features=20)

X = vectorizer.fit_transform( newsgroups_train.data + newsgroups_test.data )
X_train = X[0:num_train, :]
X_test = X[num_train:num_train+num_test,:]

Y_train = newsgroups_train.target
Y_test = newsgroups_test.target

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# Generate a dataset and plot it
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# %% 4
# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

X = [1,2,3]

num_examples = len(X) # 训练样本的数量
nn_input_dim = 2 # 输入层的维度
nn_output_dim = 2 # 输出层的维度

# 梯度下降的参数（我直接手动赋值）
epsilon = 0.01 # 梯度下降的学习率
reg_lambda = 0.01 # 正则化的强度

nn_architecture = [
    {'input_dim': nn_input_dim, 'output_dim': 4, 'actFunc': 'tanh'},
    {'input_dim': 4, 'output_dim': 6, 'actFunc': 'tanh'},
    {'input_dim': 6, 'output_dim': 6, 'actFunc': 'tanh'},
    {'input_dim': 6, 'output_dim': 4, 'actFunc': 'tanh'},
    {'input_dim': 4, 'output_dim': nn_output_dim, 'actFunc': 'tanh'},
]

def init_layer(input_dim, output_dim, actFunc):
    np.random.seed(0)
    W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
    b = np.zeros(1,output_dim)
    layer = {'W': W, 'b': b, 'actFunc': actFunc}
    return layer

def init_layers(nn_architecture):
    layers = []
    for l in nn_architecture:
        layer = init_layer(l.input_dim, l.output_dim, l.actFunc)
        layers.append(layer)
    return layers

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def softmax(Z):
    exp_scores = np.exp(Z)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

def loss(Z, y):
    # 计算损失
    probs = softmax(Z)
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    #在损失上加上正则项（可选）
    # data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1-sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def tanh_backward(dA, Z):
    t = np.tanh(Z)
    return dA * (1 - t ** 2)

def single_layer_forward_prop(X, layer):
    Z = X.dot(layer['W']) + layer['b']
    if layer.actFunc is 'relu':
        actFunction = relu
    elif layer.actFunc is 'sigmoid':
        actFunction = sigmoid
    else:
        actFunction = np.tanh
    return actFunction(Z), Z

def full_layers_forward_prop(X, layers):
    memory_forward = []
    Z_out = X
    # layers = init_layers(nn_architecture)
    for layer in layers:
        Z_out, Z_hide = single_layer_forward_prop(Z_out, layer)
        memo_forward = {
            'Z_out': Z_out,
            'Z_hide': Z_hide
        }
        memory_forward.append(memo_forward)

    # 返回最终的Z_out => actFunc(Z=X*W + b)
    # memory_forward记录每一层的Z_out=actFunc(Z_hide)和Z_hide=W*X+b
    memo_forward = {
        'Z_out': X
    }
    memory_forward.append(memo_forward)
    return Z_out, memory_forward

def single_layer_backward_prop(memo_forward_now, memo_forward_pre, dA_now, layer):
    # 前向神经元个数
    # dA_now为由下一层传回的梯度
    # memo_forward_pre 记录上一层计算结果， Z_hide=X*w+b和Z_out => X_pre
    # memo_forward_now 记录当前层的计算结果，Z_hide => Z_now和Z_out
    X_pre = memo_forward_pre.Z_out
    Z_now = memo_forward_now.Z_hide
    back_dim = X_pre.shape[0]

    if layer.actFunc is 'sigmoid':
        actFuncBack = sigmoid_backward
    elif layer.actFunc is 'relu':
        actFuncBack = relu_backward
    else:
        actFuncBack = tanh_backward

    # 计算当前层外层导数
    # dZ_now = actFunc'(Z_hide)
    dZ_now = actFuncBack(dA_now, Z_now)
    # dW_now = actFunc'(Z_hide) * (X=Z_hide*dW)
    dW_now = dZ_now.dot(X_pre.T) / back_dim
    # db_now = actFunc'(Z_hide) * (1=Z_hide*db); 维度转换
    db_now = np.sum(dZ_now, axis=1, keepdims=True) / back_dim
    # dA_pre为向前一层传递的梯度；对上一层的Z_out即本层的X求导结果
    # dA_pre = actFunc'(Z_hide) * (W=Z_hide*dX)
    dA_pre = dZ_now.dot(layer.W)
    return dA_pre, dA_now, dW_now, db_now

def full_layers_backward_prop(memory_forward, layers, X, y):
    Z_out, memo_forward = full_layers_forward_prop(X, layers)
    # dA_now = 0
    # 反向传播
    probs = softmax(Z_out)
    probs[range(num_examples), y] -= 1
    dA_now = probs
    memory_backward = []

    for idx,layer in layers.reverse():
        dA_pre, dA_now, dW_now, db_now = single_layer_backward_prop(memo_forward[idx],memo_forward[idx+1],dA_now,layer)
        memo_backward = {
            # 'dA_pre': dA_pre,
            # 'dA_now': dA_now,
            'dW_now': dW_now,
            'db_now': db_now
        }
        memory_backward.append(memo_backward)
    return memory_backward

def update(layers, memory_backward):
    for idx,layer in layers:
        layer.W -= epsilon * memory_backward[idx]['dW']
        layer.b -= epsilon * memory_backward[idx]['db']
    return layers

def predict(Z):
    probs = softmax(Z)
    return np.argmax(probs, axis=1)

def train(X, y, nn_architcture, epochs):
    layers = init_layers(nn_architecture)
    cost_history = []
    # accuracy_history = []

    for i in range(epochs):
        Z_out, memory_forward = full_layers_forward_prop(X,layers)
        cost = loss(Z_out, y)
        cost_history.append(cost)

        memory_backward = full_layers_backward_prop(memory_forward, layers, X, y)
        layers = update(layers, memory_backward)

    return layers, cost_history
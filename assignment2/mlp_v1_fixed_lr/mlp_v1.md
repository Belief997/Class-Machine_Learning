

```python
# %% 1
# Package imports
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import copy

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

newsgroups_train = fetch_20newsgroups(subset='train',  categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',  categories=categories)

# pprint(newsgroups_train.data[0])

num_train = len(newsgroups_train.data)
num_test  = len(newsgroups_test.data)

vectorizer = TfidfVectorizer(max_features=100)

X = vectorizer.fit_transform( newsgroups_train.data + newsgroups_test.data )
X_train = X[0:num_train, :]
X_test = X[num_train:num_train+num_test,:]

Y_train = newsgroups_train.target
Y_test = newsgroups_test.target


# Normalize the data: subtract the mean image
# mean_image = np.mean(X_train, axis = 0)
# X_train -= mean_image
# X_test -= mean_image

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
```

    (2034, 100) (2034,)
    (1353, 100) (1353,)
    


```python
# # Generate a dataset and plot it
# np.random.seed(0)
# X, y = sklearn.datasets.make_moons(1000, noise=0.20)
# print('输入：',X.shape)
# print('输出',y.shape)
# plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
```


```python
# # Train the logistic regression classifier
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X, y)
```


```python
# # Helper function to plot a decision boundary.
# # If you don't fully understand this function don't worry, it just generates the contour plot below.
# def plot_decision_boundary(pred_func):
#     # Set min and max values and give it some padding
#     x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#     y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#     h = 0.01
#     # Generate a grid of points with distance h between them
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     # Predict the function value for the whole gid
#     Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     # Plot the contour and training examples
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
```


```python
# # Plot the decision boundary
# plot_decision_boundary(lambda x: clf.predict(x))
# plt.title("Logistic Regression")
```

初始化层


```python
def init_layer(input_dim, output_dim, actFunc):
    np.random.seed(0)
    W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
    b = np.zeros((1,output_dim))
    print('w:',W.shape)
    print('b:',b.shape)
    layer = {'W': W, 'b': b, 'actFunc': actFunc}
    return layer
```


```python
def init_layers(nn_architecture):
    layers = []
    for l in nn_architecture:
        layer = init_layer(l['input_dim'], l['output_dim'], l['actFunc'])
        layers.append(layer)
    return layers
```

激活函数


```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

```

softmax


```python
def softmax(Z):
    exp_scores = np.exp(Z)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs
```

反向传播


```python
def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1-sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def tanh_backward(dA, Z):
    t = np.tanh(Z)
    res = (1 - t * t)
#     print('res:', res.shape)
#     print('dA:', dA.shape)
    return res * dA
```

损失函数


```python
def loss(Z, y):
    # 计算损失
    probs = softmax(Z)
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    #在损失上加上正则项（可选）
    # data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss
```

前向传播


```python
def single_layer_forward_prop(X, layer):
    W = layer['W']
#     print(W.shape)
    Z = X.dot(layer['W']) + layer['b']
    if layer['actFunc'] is 'relu':
        actFunction = relu
    elif layer['actFunc'] is 'sigmoid':
        actFunction = sigmoid
    else:
        actFunction = np.tanh
    return actFunction(Z), Z
```


```python
def full_layers_forward_prop(X, layers):
    memory_forward = []
    Z_out = X
    memo_forward = {
        'Z_out': X
    }
    memory_forward.append(memo_forward)
    layers_now = 0
    for layer in layers:
#         print('forward layers_now:',layers_now)
        Z_out, Z_hide = single_layer_forward_prop(Z_out, layer)
        memo_forward = {
            'Z_out': Z_out,
            'Z_hide': Z_hide
        }
        memory_forward.append(memo_forward)
        layers_now += 1

    # 返回最终的Z_out => actFunc(Z=X*W + b)
    # memory_forward记录每一层的Z_out=actFunc(Z_hide)和Z_hide=W*X+b
#     print('Z_out: ',Z_out.shape)
    return Z_out, memory_forward
```

反向传播


```python
def single_layer_backward_prop(memo_forward_now, memo_forward_pre, dA_now, layer):
    # 前向神经元个数
    # dA_now为由下一层传回的梯度
    # memo_forward_pre 记录上一层计算结果， Z_hide=X*w+b和Z_out => X_pre
    # memo_forward_now 记录当前层的计算结果，Z_hide => Z_now和Z_out
    X_pre = memo_forward_pre['Z_out']
    Z_now = memo_forward_now['Z_hide']
    back_dim = X_pre.shape[0]

    if layer['actFunc'] is 'sigmoid':
        actFuncBack = sigmoid_backward
    elif layer['actFunc'] is 'relu':
        actFuncBack = relu_backward
    else:
        actFuncBack = tanh_backward

    # 计算当前层外层导数
    # dZ_now = actFunc'(Z_hide)
    dZ_now = actFuncBack(dA_now, Z_now)
    # dW_now = actFunc'(Z_hide) * (X=Z_hide*dW)
#     print('X_pre',X_pre.shape)
#     print('dZ_now',dZ_now.shape)
#     print('dA_now',dA_now.shape)
#     print('Z_now',Z_now.shape)
    dW_now = X_pre.T.dot(dZ_now) / back_dim
    # db_now = actFunc'(Z_hide) * (1=Z_hide*db); 维度转换
    db_now = np.sum(dZ_now, axis=0, keepdims=True) / back_dim
#     print('dW_now:',dW_now.shape)
#     print('db_now',db_now.shape)
    # dA_pre为向前一层传递的梯度；对上一层的Z_out即本层的X求导结果
    # dA_pre = actFunc'(Z_hide) * (W=Z_hide*dX)
    W_now = copy.deepcopy(layer['W'])
    dA_pre = dZ_now.dot(W_now.T)
#     print('dA_pre',dA_pre.shape)
    
    return dA_pre,dW_now, db_now
```


```python
def full_layers_backward_prop(Z_out, memory_forward, layers, X, y):
#     Z_out, memo_forward = full_layers_forward_prop(X, layers)
    # 反向传播
    probs = softmax(Z_out)
    probs[range(num_examples), y] -= 1
    dA_pre = probs
#     print('dA_now:', dA_now.shape)
#     print('probs:', probs.shape)
    memory_backward = []
    layers.reverse()
    memory_forward.reverse()

    length = len(layers)
    for idx in range(length):
#         print('layer_now:', idx)
        dA_pre, dW_now, db_now = single_layer_backward_prop(memory_forward[idx],memory_forward[idx+1],dA_pre,layers[idx])
        memo_backward = {
            'dW_now': dW_now,
            'db_now': db_now
        }
        memory_backward.append(memo_backward)

    return memory_backward
```

更新网络


```python
def update(layers, memory_backward):
#     print('layers: ',len(layers)
#     print('memory_backward: ', len(memory_backward))
    length = len(layers)
#     print(memory_backward)
#     print(layers)
#     print(memory_backward)
    for idx in range(length):
        dW = memory_backward[idx]['dW_now']
#         print('dW.shape: ', dW.shape)
        layers[idx]['W'] -= epsilon * memory_backward[idx]['dW_now']
        layers[idx]['b'] -= epsilon * memory_backward[idx]['db_now']
        
#     print(memory_backward)
#     print(layers)
    return layers
```

预测函数


```python
def predict(X, layers):
    Z_out, memory_forward = full_layers_forward_prop(X,layers)
    probs = softmax(Z_out)
    return np.argmax(probs, axis=1)
```

计算准确率


```python
def get_acc(X, layers):
    Z_out, memory_forward = full_layers_forward_prop(X,layers)
    probs = softmax(Z_out)
#     print(probs)
#     return np.argmax(probs, axis=1)
    acc = np.mean(Y_test==np.argmax(probs, axis=1))
    return acc
```

训练函数


```python
def train(X, y, nn_architcture, epochs):
    layers = init_layers(nn_architcture)
    cost_history = []
    accuracy_history = []
    best_acc = 0

    for i in range(epochs):
        Z_out, memory_forward = full_layers_forward_prop(X,layers)
#         print(Z_out.shape)
        cost = loss(Z_out, y)
        acc = get_acc(X_test, layers)
        cost_history.append(cost)
        accuracy_history.append(acc)
        if best_acc < acc :
            best_acc = acc
            
        if i % 500 == 0:
#             epsilon = 0.01 * np.exp(-i / 500) + 0.02
            print('||best_acc => ', best_acc, '||cost => ', cost, '||acc => ', acc)
#             print('epochs: ', i, 'epsilon: ', epsilon)
#             print('cost: ', cost)
#             print('acc: ', acc)


        memory_backward = full_layers_backward_prop(Z_out, memory_forward, layers, X, y)
        layers = update(layers, memory_backward)
        
        layers.reverse()
        memory_forward.reverse()
    
    plt.plot(accuracy_history)
    plt.ylabel('accuracy')
    plt.show()
    
    
    plt.plot(cost_history)
    plt.ylabel('loss')
    plt.show()
#     plt.title('the net layers: ', len(nn_architcture))
    
    return layers, cost_history, accuracy_history
```

初始化参数，网络结构，学习率


```python
num_examples = X_train.shape[0] # 训练样本的数量
nn_input_dim = X_train.shape[1] # 输入层的维度
nn_output_dim = 4 # 输出层的维度

# 梯度下降的参数（我直接手动赋值）
epsilon = 0.05 # 梯度下降的学习率
reg_lambda = 0.01 # 正则化的强度
epochs = 5000

print('样本数量：', num_examples)
print('输入样本维度：',nn_input_dim)
print('输出数量：',nn_output_dim)
```

    样本数量： 2034
    输入样本维度： 100
    输出数量： 4
    

输出


```python
# nn_architcture = [
# #     {'input_dim': nn_input_dim, 'output_dim': 4, 'actFunc': 'tanh'},
# #     {'input_dim': 4, 'output_dim': 6, 'actFunc': 'tanh'},
# #     {'input_dim': 6, 'output_dim': 6, 'actFunc': 'tanh'},
# #     {'input_dim': 6, 'output_dim': 4, 'actFunc': 'tanh'},
#     {'input_dim': nn_input_dim, 'output_dim': nn_output_dim, 'actFunc': 'tanh'},
# ]

# layers, cost_history,accuracy_history = train(X_train, Y_train, nn_architcture, epochs)
# acc = get_acc(X_test, layers)
# # plot_decision_boundary(lambda x: predict(x, layers))
# # plt.title("Decision Boundary for hidden layer size 1.")
# print("Decision Boundary for hidden layer size 1. acc: ", acc)
```


```python
# nn_architcture = [
#     {'input_dim': nn_input_dim, 'output_dim': 18, 'actFunc': 'tanh'},
# #     {'input_dim': 4, 'output_dim': 6, 'actFunc': 'tanh'},
# #     {'input_dim': 6, 'output_dim': 6, 'actFunc': 'tanh'},
# #     {'input_dim': 6, 'output_dim': 4, 'actFunc': 'tanh'},
#     {'input_dim': 18, 'output_dim': nn_output_dim, 'actFunc': 'tanh'},
# ]

# layers, cost_history,accuracy_history  = train(X_train, Y_train, nn_architcture, epochs)
# acc = get_acc(X_test, layers)
# # plot_decision_boundary(lambda x: predict(x, layers))
# # plt.title("Decision Boundary for hidden layer size 2.")
# print("Decision Boundary for hidden layer size 2, acc: ", acc)
```


```python
nn_architcture = [
    {'input_dim': nn_input_dim, 'output_dim': 18, 'actFunc': 'tanh'},
    {'input_dim': 18, 'output_dim': 18, 'actFunc': 'tanh'},
#     {'input_dim': 20, 'output_dim': 20, 'actFunc': 'tanh'},
#     {'input_dim': 20, 'output_dim': 8, 'actFunc': 'tanh'},
    {'input_dim': 18, 'output_dim': nn_output_dim, 'actFunc': 'tanh'},
]

layers, cost_history,accuracy_history  = train(X_train, Y_train, nn_architcture, epochs)
acc = get_acc(X_test, layers)
# plot_decision_boundary(lambda x: predict(x, layers))
# plt.title("Decision Boundary for hidden layer size 3")
print("Decision Boundary for hidden layer size 3. acc: ", acc)
```

    w: (100, 18)
    b: (1, 18)
    w: (18, 18)
    b: (1, 18)
    w: (18, 4)
    b: (1, 4)
    ||best_acc =>  0.21729490022172948 ||cost =>  1.397058409612424 ||acc =>  0.21729490022172948
    ||best_acc =>  0.6186252771618626 ||cost =>  1.0215821967650556 ||acc =>  0.6186252771618626
    ||best_acc =>  0.6511456023651145 ||cost =>  0.8489406972354088 ||acc =>  0.6459719142645972
    ||best_acc =>  0.6541019955654102 ||cost =>  0.7979878945449932 ||acc =>  0.6533628972653363
    ||best_acc =>  0.6600147819660015 ||cost =>  0.773308828033585 ||acc =>  0.6600147819660015
    ||best_acc =>  0.6629711751662971 ||cost =>  0.7571158245214179 ||acc =>  0.6600147819660015
    ||best_acc =>  0.6629711751662971 ||cost =>  0.7447845300391851 ||acc =>  0.6614929785661493
    ||best_acc =>  0.6629711751662971 ||cost =>  0.7347234599850934 ||acc =>  0.6622320768662232
    ||best_acc =>  0.6666666666666666 ||cost =>  0.7263143356098677 ||acc =>  0.6659275683665927
    ||best_acc =>  0.6666666666666666 ||cost =>  0.7191913713682916 ||acc =>  0.6651884700665188
    


![png](output_35_1.png)



![png](output_35_2.png)


    Decision Boundary for hidden layer size 3. acc:  0.663710273466371
    


```python
nn_architcture = [
    {'input_dim': nn_input_dim, 'output_dim': 20, 'actFunc': 'tanh'},
    {'input_dim': 20, 'output_dim': 16, 'actFunc': 'tanh'},
#     {'input_dim': 16, 'output_dim': 16, 'actFunc': 'tanh'},
    {'input_dim': 16, 'output_dim': 20, 'actFunc': 'tanh'},
    {'input_dim': 20, 'output_dim': nn_output_dim, 'actFunc': 'tanh'},
]

layers, cost_history,accuracy_history  = train(X_train, Y_train, nn_architcture, epochs)
acc = get_acc(X_test, layers)
# plot_decision_boundary(lambda x: predict(x, layers))
# plt.title("Decision Boundary for hidden layer size 4")
print("Decision Boundary for hidden layer size 4. acc: ", acc)
```

    w: (100, 20)
    b: (1, 20)
    w: (20, 16)
    b: (1, 16)
    w: (16, 20)
    b: (1, 20)
    w: (20, 4)
    b: (1, 4)
    ||best_acc =>  0.3887657058388766 ||cost =>  1.366441511813068 ||acc =>  0.3887657058388766
    ||best_acc =>  0.6334072431633407 ||cost =>  0.8861130162311656 ||acc =>  0.6334072431633407
    ||best_acc =>  0.6504065040650406 ||cost =>  0.7936117294442915 ||acc =>  0.647450110864745
    ||best_acc =>  0.6511456023651145 ||cost =>  0.7652458603523965 ||acc =>  0.6481892091648189
    ||best_acc =>  0.6511456023651145 ||cost =>  0.7496858583032696 ||acc =>  0.6504065040650406
    ||best_acc =>  0.6577974870657798 ||cost =>  0.7384196972100668 ||acc =>  0.6555801921655581
    ||best_acc =>  0.6577974870657798 ||cost =>  0.7283499436683212 ||acc =>  0.6548410938654841
    ||best_acc =>  0.6577974870657798 ||cost =>  0.7188230857750735 ||acc =>  0.6541019955654102
    ||best_acc =>  0.6577974870657798 ||cost =>  0.7154787274595216 ||acc =>  0.6518847006651884
    ||best_acc =>  0.6585365853658537 ||cost =>  0.7071402006518752 ||acc =>  0.6511456023651145
    


![png](output_36_1.png)



![png](output_36_2.png)


    Decision Boundary for hidden layer size 4. acc:  0.6496674057649667
    


```python
nn_architcture = [
    {'input_dim': nn_input_dim, 'output_dim': 20, 'actFunc': 'tanh'},
    {'input_dim': 20, 'output_dim': 16, 'actFunc': 'tanh'},
    {'input_dim': 16, 'output_dim': 16, 'actFunc': 'tanh'},
    {'input_dim': 16, 'output_dim': 20, 'actFunc': 'tanh'},
    {'input_dim': 20, 'output_dim': nn_output_dim, 'actFunc': 'tanh'},
]

layers, cost_history,accuracy_history  = train(X_train, Y_train, nn_architcture, epochs)
acc = get_acc(X_test, layers)
# plot_decision_boundary(lambda x: predict(x, layers))
# plt.title("Decision Boundary for hidden layer size 4")
print("Decision Boundary for hidden layer size 4. acc: ", acc)
```

    w: (100, 20)
    b: (1, 20)
    w: (20, 16)
    b: (1, 16)
    w: (16, 16)
    b: (1, 16)
    w: (16, 20)
    b: (1, 20)
    w: (20, 4)
    b: (1, 4)
    ||best_acc =>  0.31042128603104213 ||cost =>  1.3902958425464178 ||acc =>  0.31042128603104213
    ||best_acc =>  0.6452328159645233 ||cost =>  0.8450045886099727 ||acc =>  0.6437546193643755
    ||best_acc =>  0.6533628972653363 ||cost =>  0.7698012436082868 ||acc =>  0.6511456023651145
    ||best_acc =>  0.6533628972653363 ||cost =>  0.757514184995692 ||acc =>  0.6481892091648189
    ||best_acc =>  0.6533628972653363 ||cost =>  0.7685013955432677 ||acc =>  0.6481892091648189
    ||best_acc =>  0.6541019955654102 ||cost =>  0.7522418975600225 ||acc =>  0.6444937176644494
    ||best_acc =>  0.6541019955654102 ||cost =>  0.7486007138034552 ||acc =>  0.647450110864745
    ||best_acc =>  0.6541019955654102 ||cost =>  0.7417111099986754 ||acc =>  0.6496674057649667
    ||best_acc =>  0.6541019955654102 ||cost =>  0.7332303310566726 ||acc =>  0.6467110125646711
    ||best_acc =>  0.6541019955654102 ||cost =>  0.7301689698008248 ||acc =>  0.6504065040650406
    


![png](output_37_1.png)



![png](output_37_2.png)


    Decision Boundary for hidden layer size 4. acc:  0.6467110125646711
    


```python
nn_architcture = [
    {'input_dim': nn_input_dim, 'output_dim': 90, 'actFunc': 'tanh'},
    {'input_dim': 90, 'output_dim': 80, 'actFunc': 'tanh'},
    {'input_dim': 80, 'output_dim': 70, 'actFunc': 'tanh'},
    {'input_dim': 70, 'output_dim': 60, 'actFunc': 'tanh'},
    {'input_dim': 60, 'output_dim': 50, 'actFunc': 'tanh'},
    {'input_dim': 50, 'output_dim': 25, 'actFunc': 'tanh'},
    {'input_dim': 25, 'output_dim': 15, 'actFunc': 'tanh'},
    {'input_dim': 15, 'output_dim': nn_output_dim, 'actFunc': 'tanh'},
]

layers, cost_history,accuracy_history  = train(X_train, Y_train, nn_architcture, epochs)
acc = get_acc(X_test, layers)
# plot_decision_boundary(lambda x: predict(x, layers))
# plt.title("Decision Boundary for hidden layer size 4")
print("Decision Boundary for hidden layer size 8. acc: ", acc)
```

    w: (100, 90)
    b: (1, 90)
    w: (90, 80)
    b: (1, 80)
    w: (80, 70)
    b: (1, 70)
    w: (70, 60)
    b: (1, 60)
    w: (60, 50)
    b: (1, 50)
    w: (50, 25)
    b: (1, 25)
    w: (25, 15)
    b: (1, 15)
    w: (15, 4)
    b: (1, 4)
    ||best_acc =>  0.29859571322985956 ||cost =>  1.3825891332546705 ||acc =>  0.29859571322985956
    ||best_acc =>  0.6504065040650406 ||cost =>  0.8070939153743654 ||acc =>  0.6356245380635624
    ||best_acc =>  0.6548410938654841 ||cost =>  0.7604786295724663 ||acc =>  0.6363636363636364
    ||best_acc =>  0.6548410938654841 ||cost =>  0.7482322065987868 ||acc =>  0.6430155210643016
    ||best_acc =>  0.6629711751662971 ||cost =>  0.7096500635388996 ||acc =>  0.6518847006651884
    ||best_acc =>  0.6651884700665188 ||cost =>  0.691440180746688 ||acc =>  0.6526237989652623
    ||best_acc =>  0.6703621581670363 ||cost =>  0.6847385768272123 ||acc =>  0.6555801921655581
    ||best_acc =>  0.6703621581670363 ||cost =>  0.6745375966250616 ||acc =>  0.6526237989652623
    ||best_acc =>  0.672579453067258 ||cost =>  0.6586085598331861 ||acc =>  0.6489283074648928
    ||best_acc =>  0.672579453067258 ||cost =>  0.6613063888665515 ||acc =>  0.6555801921655581
    


![png](output_38_1.png)



![png](output_38_2.png)


    Decision Boundary for hidden layer size 8. acc:  0.6496674057649667
    

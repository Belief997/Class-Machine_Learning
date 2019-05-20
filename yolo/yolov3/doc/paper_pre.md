## 原理理解

yolov3结构框图

<img src='./yolov3/yolov3_structure.jpg'>



## 源码阅读

### 数据读取部分

1. 文件组织架构

2. 下载数据集

3. 配置文件

   ```python
   [convolutional] #卷积层
   batch_normalize=1
   filters=32
   size=3
   stride=1
   pad=1
   activation=leaky
   
   [shortcut] #跳过连接，与resnet类似，表示当前输出个前第三层输出与模块输入相加
   from=-3
   activation=linear

   [yolo] #检测层，三个尺寸分别3个anchor，共九个anchor，mask有3个表示3个尺度，设置阈值和类别数
   mask = 6,7,8
   anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
   classes=80
   num=9
   jitter=.3
   ignore_thresh = .7
   truth_thresh = 1
   random=1

   [route] #route可能有一个值，可能有两个值（-1,61）为-4时表示将输出前第四层的特征图，为-1和61时表示将输出前一层与第61层特征图的拼接结果
   layers = -4

   [upsample] #上采样，三个尺度，步幅分别为32,16,8，通过上采样实现
   stride=2
   ```

4. 数据读取

### 模型训练部分

1. 基本流程

   加载模型 -> 初始化数据 -> 预定义优化器 -> 训练

2. 模型

3. yolo层

   **该层对应的是网络的最后一层(y1,y2,y3).首先获得预测结果prediction(x,y,w,h,con,cls).**

   再计算网格单元左上角坐标和锚节点对应比例,这个锚节点是聚类计算过的大小,大小固定,所以直接可以使用.

   在通过相对坐标和偏移量计算实际坐标.锚节点就是下列公式中的pw和ph。

   <img src="./yolov3/坐标转换公式.png">

   再计算真值标签相对于gird的真值标签.

   最后计算损失，在计算损失中只计算真值与锚节点重合面积最大的检测窗口的损失（每一个网格预测有三个预测结果，每一个对应一个锚节点，只计算锚节点和真值IOU最大的那个预测结果的损失）

4. 总网络

   卷积层+cat连接层+点加层+输出层

### 测试部分

1. recall-precious

2. ap的计算

   <img src='./yolov3/mAP含义.png'>

   <img src='./yolov3/mAP计算.png'>


# yolo与ssd

本次试验，我们尝试使用SSD来进行物体检测。

## SSD: Single Shot Multibox Detector

论文连接：[SSD: Single Shot Multibox Detector](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1512.02325)

参考连接：[SSD](https://zhuanlan.zhihu.com/p/24954433)

参考PPT：[SSD PPT](https://docs.google.com/presentation/d/1rtfeV_VmdGdZD5ObVVpPDPIODSDxKnFSU0bsN_rgZXc/pub?start=false&loop=false&delayms=3000&slide=id.g179f601b72_0_51)

对比yolo：[yolo对比ssd](<http://lanbing510.info/2017/08/28/YOLO-SSD.html>)



![preview](https://pic2.zhimg.com/v2-32884ad14bc74f5cc17b17ffccb0e2f5_r.jpg)

<center>SSD网络结构</center>

SSD相比YOLO有以下突出的特点：

- 多尺度的feature map：基于VGG的不同卷积段，输出feature map到回归器中。这一点试图提升小物体的检测精度。
- 更多的anchor box，每个网格点生成不同大小和长宽比例的box，并将类别预测概率基于box预测（YOLO是在网格上），得到的输出值个数为(C+4)×k×m×n，其中C为类别数，k为box个数，m×n为feature map的大小。


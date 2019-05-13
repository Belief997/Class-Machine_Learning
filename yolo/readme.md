## yolo物体检测

### 开题
本文尝试用yolov3进行物体检测，并将尽力对比yolo各个版本在物体检测的精度和速度上的差异，同时通过阅读现有的物体检测方案，给出一定程度的优化方案和实践结果。

[yolov3论文链接](https://pjreddie.com/media/files/papers/YOLOv3.pdf)	[github地址](<https://github.com/eriklindernoren/PyTorch-YOLOv3>)

[yolo9000](https://arxiv.org/abs/1612.08242)

[yolo论文链接](https://arxiv.org/abs/1506.02640)

- yolov3

  包含pytorch实现yolov3源码，具体使用阅读`PyTorch-YOLOv3-master`文件中`readme.md`文件

  data文件夹的运行`bash get_tiny_vid.sh`即可下载并解压数据集，mac与linux用户可自行修改内容

  ```shell
  # windows
  curl -O https://xinggangw.info/data/tiny_vid.zip
  unzip -q tiny_vid.zip
  
  # #mac or linux
  # wget -c https://xinggangw.info/data/tiny_vid.zip
  # unzip -q tiny_vid.zip
  ```

  


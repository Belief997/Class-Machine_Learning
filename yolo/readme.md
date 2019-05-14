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

  

.
├── LICENSE
├── README.md
├── assets
│   ├── dog.png
│   ├── giraffe.png
│   ├── messi.png
│   └── traffic.png
├── config
│   ├── coco.data
│   ├── create_custom_model.sh
│   ├── custom.data
│   ├── yolov3-tiny.cfg
│   └── yolov3.cfg
├── data
│   ├── coco.names
│   ├── custom
│   │   ├── classes.names
│   │   ├── images
│   │   │   └── train.jpg
│   │   ├── labels
│   │   │   └── train.txt
│   │   ├── train.txt
│   │   └── valid.txt
│   ├── get_coco_dataset.sh
│   └── samples
│       ├── dog.jpg
│       ├── eagle.jpg
│       ├── field.jpg
│       ├── giraffe.jpg
│       ├── herd_of_horses.jpg
│       ├── messi.jpg
│       ├── person.jpg
│       ├── room.jpg
│       └── street.jpg
├── detect.py
├── models.py
├── requirements.txt
├── test.py
├── train.py
├── utils
│   ├── __init__.py
│   ├── augmentations.py
│   ├── datasets.py
│   ├── logger.py
│   ├── parse_config.py
│   └── utils.py
└── weights
    └── download_weights.sh
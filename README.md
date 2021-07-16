本项目为我给pytorch的dataloader（数据加载+预处理）提速的代码

具体内容参考博客《[【pytorch】给dataloader提速：Kornia库 + GPU](https://blog.csdn.net/OTZ_2333/article/details/118655925)》

`data_loaders_origin.py`为来自STFAN的原始dateloader，`data_loaders_kornia.py`为我是用Kornia库提速后的dataloader。两个py文件均可直接运行，运行参数在`config.py`中。数据集可以是用[DVD数据集](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset.zip)，或者目录类似的数据集。目录格式为

```
.
├── test
│   ├── 720p_240fps_2
│   │   ├── GT
│   │   │   ├── 00000.jpg
│   │   │   ├── 00001.jpg
│   │   │   ├── 00002.jpg
│   │   │   └── ...
│   │   └── input
│   │       ├── 00000.jpg
│   │       └── ...
│   └── IMG_0003
│       ├── GT
│       │   └── ...
│       └── input
│           └── ...
└── train
    ├── 720p_240fps_1
    │   ├── GT
    │   └── input
    └── ...
```


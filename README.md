# YOLOv3_ReSAM
YOLOv3_ReSAM:A Small Target Detection Method With Spatial Attention  Module

### 准备工作
* 推荐使用**Ubuntu 18.04**
* **CMake >= 3.8**: https://cmake.org/download/
* **CUDA >= 10.0**: https://developer.nvidia.com/cuda-toolkit-archive
* **OpenCV >= 2.4**: https://opencv.org/releases.html
* **cuDNN >= 7.0 for CUDA >= 10.0** https://developer.nvidia.com/rdp/cudnn-archive
* **GPU with CC >= 3.0**: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
* **GCC**

### Linux上编译
下载YOLOv4源码，推荐使用**Ubuntu 18.04**：
```
sudo apt-get install -y git
git clone https://github.com/AlexeyAB/darknet.git
```
> 配置`Makefile`⽂件中的参数，然后运⾏`make -j8`进⾏编译，具体参数解释如下：
* `GPU=1` 使用CUDA和GPU（CUDA默认路径为`/usr/local/cuda`）
* `CUDNN=1`使用cuDNN v5-v7加速网络（cuDNN默认路径`/usr/local/cudnn`）
* `CUDNN_HALF=1` 使用Tensor Cores（可用GPU为Titan V / Tesla V100 / DGX-2或者更新的）检
测速度3x，训练速度2x
* `OPENCV=1` 使用OpenCV 4.x/3.x/2.4.x，运⾏检测视频和摄像机
* `DEBUG=1` 编译调试版本
* `OPENMP=1` 使用OpenMP利用多CPU加速
* `LIBSO=1` 编译`darknet.so`
* 使用`uselib`来运⾏YOLO，输⼊指令如下：
`LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib test.mp4`

###在自己修改或添加的模块代码中嵌⼊YOLOv3中

### 预训练模型
所有模型都是在VisDrone2019-DET数据集上训练，模型包括两个⽂件（`cfg`和`weights`）

### 运⾏指令介绍
需要将训练好的`weights`⽂件放到`darknet`根目录下，运⾏如下指令：
* 检测单张图像
```
./darknet detector test cfg/uav.data cfg/yolov3_ReSAM.cfg yolov3_ReSAM.weights -thresh 0.25
```
* 检测给定路径的单张图像（参数最后的路径需要写待检测图像的路径）
```
./darknet detector test cfg/uav.data cfg/yolov3_ReSAM.cfg yolov3_ReSAM.weights -ext_output /home/jario/Pictures/h1.jpg
```
* 检测给定路径的单个视频
```
./darknet detector demo cfg/uav.data cfg/yolov3_ReSAM.cfg yolov3_ReSAM.weights -ext_outputtest.mp4
```
* 检测给定路径的单个视频，并将检测结果保存为视频
```

```
### 开始训练模型

(1) 针对选择的模型，下载预训练权重：
* 对于`yolov3.cfg, yolov3-spp.cfg` (154 MB): [darknet53.conv.74](https://
pjreddie.com/media/files/darknet53.conv.74)
* 对于`yolov3-tiny-prn.cfg , yolov3-tiny.cfg` (6 MB): [yolov3-tiny.conv.11]
(https://drive.google.com/file/d/18v36esoXCh-PsOKwyP2GWrpYDptDY8Zf/view?usp=sharing)

(2)在`darknet/data`路径下创建`uav.names`，其中每一⾏是一个目标类别名称
* 将数据集标注得到的⽂件重命名为`uav.names`，并放到`darknet/data`下

（3） 在`darknet/data`路径下创建`obj.data`：
* 教程darknet 路径为`/home/user/darknet`，本⽂以此为例，请根据自己的路径进⾏修改。
在/home/user/darknet/cfg/ ⽂件夹下新建一个⽂件，名字叫obj.data 在⾥面写⼊：
```
classes = 10
train = /home/user/darknet/data/coco/visdrone_20180908_234114.txt
valid = /home/user/darknet/data/coco/visdrone_20180908_234114.txt
names = data/uav.names
backup = backup
eval = coco
```
> 注意：classes 为类别数量，对于visdrone数据集10分类目标检测问题，写10

（4）将visdrone2019数据集中图像⽂件（.jpg）与标注⽂件放⼊到如下路径`darknet\data\coco\`路径下

(5)开始训练
* 训练指令：`./darknet detector train data/obj.data cfg/yolov3_ReSAM.cfg yolov3.conv.53`
* （对于最新100次迭代的最新权重`yolo-obj_last.weights`会保存在`darknet\backup\`）
* （对于每1000次迭代的权重`yolo-obj_xxxx.weights`会保存在`darknet\backup\`）
* （关闭Loss的显示窗口`./darknet detector train data/obj.data cfg/yolov3_ReSAM.cfg yolov3.conv.53 -dont_show`）
* （通过浏览器查看训练过程`./darknet detector train data/obj.data cfg/yolov3_ReSAM.cfg yolov3.conv.53 -dont_show -mjpeg_port 8090 -map`，然后打开Chrome浏览器，输⼊`http://ip-address:8090`）
* （如果需要在训练中计算mAP，每4期计算一次，需要在`obj.data`⽂件中设置
`valid=valid.txt`，运⾏：`./darknet detector train data/obj.data cfg/yolov3_ReSAM.cfg yolov3.conv.53 -map`）
7. 训练结束，结果保存在`darknet\backup\yolo-obj_final.weights`
* 如果训练中断，可以选择一个保存的权重继续训练，使用`./darknet detector train data/obj.data yolov3_ReSAM.cfg backup\yolo-obj_2000.weights`
> 注意：在训练中，如果`avg`（loss）出现`nan`，则训练出了问题，

(6)当训练停止之后，可以从`darknet\backup`中取出最新保存的训练权重`.weights`，并选择它们中检测效果最好的
例如，当训练9000次停止后，效果最好的模型可能是之前保存权重中的一个（7000,8000,9000），这是因为过拟合（Overfiting）现象。过拟合的表现可以解释为，在训练图像上检测效果很好，但是在其他图像上效果不佳，这时候就该尽早停止训练（**早停点**）。

(7)运⾏训练好的模型，进⾏目标检测，执⾏：
```
./darknet detector test data/obj.data cfg/yolov3_ReSAM.cfg yolov3_ReSAM_300000.weights
```

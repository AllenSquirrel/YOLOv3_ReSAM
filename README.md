# YOLOv3_ReSAM
YOLOv3_ReSAM:A Small Target Detection Method With Residual Spatial Attention  Module

### 准备工作
* 推荐使用**Ubuntu 18.04**
* **CMake >= 3.8**: https://cmake.org/download/
* **CUDA >= 10.0**: https://developer.nvidia.com/cuda-toolkit-archive
* **OpenCV >= 2.4**: https://opencv.org/releases.html
* **cuDNN >= 7.0 for CUDA >= 10.0** https://developer.nvidia.com/rdp/cudnn-archive
* **GPU with CC >= 3.0**: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
* **GCC**
* 
### VisDrone数据集下载
*下载地址：
* http://aiskyeye.com/download/object-detection-2/
 
Drones, or general UAVs, equipped with cameras have been fast deployed to a wide range of applications, including agricultural, aerial photography, fast delivery, and surveillance. Consequently, automatic understanding of visual data collected from these platforms become highly demanding, which brings computer vision to drones more and more closely. We are excited to present a large-scale benchmark with carefully annotated ground-truth for various important computer vision tasks, named VisDrone, to make vision meet drones.

The VisDrone2021 dataset is collected by the AISKYEYE team at Lab of Machine Learning and Data Mining , Tianjin University, China. The benchmark dataset consists of 400 video clips formed by 265,228 frames and 10,209 static images, captured by various drone-mounted cameras, covering a wide range of aspects including location (taken from 14 different cities separated by thousands of kilometers in China), environment (urban and country), objects (pedestrian, vehicles, bicycles, etc.), and density (sparse and crowded scenes). Note that, the dataset was collected using various drone platforms (i.e., drones with different models), in different scenarios, and under various weather and lighting conditions. These frames are manually annotated with more than 2.6 million bounding boxes or points of targets of frequent interests, such as pedestrians, cars, bicycles, and tricycles. Some important attributes including scene visibility, object class and occlusion, are also provided for better data utilization.

VisDrone2019 数据集由中国天津大学机器学习与数据挖掘实验室的 AISKYEYE 团队发布。基准数据集由265228帧和10209张静态图像组成的400个视频片段组成，涵盖范围广泛，包括位置（取自中国相隔数千公里的不同地域）、环境（城镇街道和农村街道）、目标（行人、三轮车、卡车等）和密度（稀疏和拥挤的场景）。此数据集是使用各种无人机平台（即具有不同型号的无人机）、在不同场景下以及在各种天气和光照条件下采集的。数据集全部是由团队手动注释的260万个边界框或感兴趣的目标点，例如行人、卡车和公交车等10类常见地面道路可移动目标。

与Microsoft COCO（MS COCO）数据集比较，单张图像最大检测目标数为100，而VisDrone2019 数据集单张图像最大检测数为500，且大部分集中在小目标上。MS COCO数据集评价指标以平均精确度AP进行评价，选择不同阈值下AP和不同目标像素面积下AP作为评价指标，且阈值通常选择为0.75。而VisDrone2019 数据集中由于小目标所占据像素面积较小，导致真实框与预测框交叉重叠率太小而判断错误。考虑到小目标的预测框太小，IOU阈值设定为0.5。因此，选择以阈值为0.5以上的AP作为评价指标，后续实验数据均以VisDrone2019 数据集阈值为0.5以上的AP作为评价指标进行实验数据对比分析。


### Linux上编译
下载DarkNet（yolov3）源码，推荐使用**Ubuntu 18.04**：
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
```
主要研究了低空无人机对地小目标检测算法

(1)扩充了特征金字塔并行结构融合了浅层特征,从而提取浅层网络模型中更多小目标细粒度特征信息到顶层特征图中进行预测。
```
![image](https://user-images.githubusercontent.com/28772715/164364533-cd6dcdcd-a4d4-4564-9964-d72cf957d1a2.png)

```
(2)多级并行特征金字塔结构中由于多层次和多尺度操作引起的重新缩放操作造成空间信息层级化的丢失，通过注入空间注意力机制弥补像素级分类的精度损失，弱化背景特征的同时强化小目标特征响应。
```
![image](https://user-images.githubusercontent.com/28772715/164364618-371e1cc4-24bd-4f4a-8137-5ffc642311ea.png)


```
(3)将空间注意力模块直接纵向加入原生主干网络结构中，新模块的添加会增加网络深度，深层的网络结构能够提取到具有丰富语义信息的抽象特征，但单一的纵向增加网络层数存在梯度消失的隐患，所以通过融入残差网络来横向加深网络。
```
![image](https://user-images.githubusercontent.com/28772715/164364648-a2b30d2a-1082-420b-ab60-85c256972f8c.png)

```
(4)针对One-Stage目标检测算法中边界回归精度低的问题，提出了基于奖赏机制的边界回归模型。引入强化学习的思想在原生边界回归策略的粗定位基础上指导边界框回归，采用变体IoU计算方式作为奖赏函数的评价指标进行精细调整。
```
![image](https://user-images.githubusercontent.com/28772715/164365031-73e830cd-b5f5-4c97-ae15-ff742cb91684.png)

```
(5)通过采用深度可分离卷积的思想对残差结构中部分标准卷积进行替换，从而减少卷积层的运算量，同时减少额外引入空间注意力机制模块的参数量。为了验证其轻量化后网络模型的性能，分别选取其IoU值为0.5下各类目标的AP大小，模型运算量BFLOPs和模型推理延迟进行实验。
```
![image](https://user-images.githubusercontent.com/28772715/165009643-d0cf2e2d-b44b-444b-94bd-ca6c226fa97c.png)

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
./darknet detector test cfg/uav.data cfg/yolov3_ReSAM.cfg yolov3_ReSAM.weights data/s2.jpg
```
![image](https://user-images.githubusercontent.com/28772715/163293284-6b557fd6-5a71-4032-b3ce-ac4c5f861272.png)

* 检测给定路径的单个视频
```
./darknet detector demo cfg/uav.data cfg/yolov3_ReSAM.cfg yolov3_ReSAM.weights data/test.mp4
```
![QQ视频20220414091941](https://user-images.githubusercontent.com/28772715/163295930-0d259155-4ea4-4d87-90ac-0a6b759005ee.gif)

* 利用摄像机实时检测
```
./darknet detector demo cfg/uav.data cfg/yolov3_ReSAM.cfg yolov3_ReSAM.weights -c 0
```
![QQ视频20220414092905](https://user-images.githubusercontent.com/28772715/163296209-6c8c169f-ab39-4c03-bc5b-b779358ff631.gif)

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
* 训练指令：`./darknet detector train data/obj.data cfg/yolov3_ReSAM.cfg darknet53.conv.74`
* （对于最新100次迭代的最新权重`yolo-obj_last.weights`会保存在`darknet\backup\`）
* （对于每1000次迭代的权重`yolo-obj_xxxx.weights`会保存在`darknet\backup\`）
* （关闭Loss的显示窗口`./darknet detector train data/obj.data cfg/yolov3_ReSAM.cfg darknet53.conv.74 -dont_show`）
* （通过浏览器查看训练过程`./darknet detector train data/obj.data cfg/yolov3_ReSAM.cfg darknet53.conv.74 -dont_show -mjpeg_port 8090 -map`，然后打开Chrome浏览器，输⼊`http://ip-address:8090`）
* （如果需要在训练中计算mAP，每4期计算一次，需要在`obj.data`⽂件中设置
`valid=valid.txt`，运⾏：`./darknet detector train data/obj.data cfg/yolov3_ReSAM.cfg darknet53.conv.74 -map`）
7. 训练结束，结果保存在`darknet\backup\yolo-obj_final.weights`
* 如果训练中断，可以选择一个保存的权重继续训练，使用`./darknet detector train data/obj.data yolov3_ReSAM.cfg backup\yolo-obj_2000.weights`
> 注意：在训练中，如果`avg`（loss）出现`nan`，则训练出了问题，

(6)当训练停止之后，可以从`darknet\backup`中取出最新保存的训练权重`.weights`，并选择它们中检测效果最好的
例如，当训练9000次停止后，效果最好的模型可能是之前保存权重中的一个（7000,8000,9000），这是因为过拟合（Overfiting）现象。过拟合的表现可以解释为，在训练图像上检测效果很好，但是在其他图像上效果不佳，这时候就该尽早停止训练（**早停点**）。

(7)运⾏训练好的模型，进⾏目标检测，执⾏：
```
./darknet detector demo cfg/uav_dataset.data cfg/yolov3_ReSAM.cfg yolov3-tiny-resnet02_120000_300000.weights test.mp4
```

### 实验结果

实验结果:
```
	                             表1  VisDrone2019-DET-Test	(不同网络模型下各类目标的平均精度均值)
             pedestrain	  people  bicycle  car	   van	   trunk    tricycle    Awing-tricycle	 bus	    motor        mAP0.5
yolov3       5.52%	  5.66%	  25.19%   10.21%  43.83%  25.64%   16.75%	14.22%	         34.85%	    38.96%	 22.08%
yolov3_ReSAM 7.28%	  5.78%	  28.70%   14.31%  53.77%  42.96%   30.48%	24.38%	         62.35%	    61.57%	 33.15%
```
准确度曲线结果：

![image](https://user-images.githubusercontent.com/28772715/163297087-893c2be5-aee3-4e67-85f8-7729ee9cc678.png)

![image](https://user-images.githubusercontent.com/28772715/163297109-1f16c37e-de1e-4f86-965d-01aa6fdaf09b.png)

```
            表2 VisionDrone2019-Changle比赛结果
排名	  模型&方法	    mAP0.5平均精度均值(%)	    AR500召回率(%)
1	      DPNetV3	          37.37	                    53.78
2	      SMPNet	          35.98	                    53.33
3	      DBNet	          35.73	                    52.57
4	      CDNet	          34.19	                    49.57
5	   ECascade R-CNN	  34.09	                    51.61
6	   FPAFS-CenterNet	  32.34	                    39.48
7	   DOHR-RetinalNet	  21.68	                    27.63
	   YOLOV3-ReSAM	          33.15%	            45.57
```
边界回归IoU曲线：

![image](https://user-images.githubusercontent.com/28772715/163297214-50a03c38-cf62-487a-bf17-5f130e779feb.png)

边界回归可视化效果：

![image](https://user-images.githubusercontent.com/28772715/163297252-d5790a13-b022-44b8-bc8c-a0c3e4b93cbd.png)

改进后的网络模型平均精度均值mAP较原生网络模型提高11.07%，且单张图像平均召回率稳定在45%左右。另一方面，建立基于奖赏机制的边界回归策略,对原生网络模型中的边界回归粗定位基础上引入强化学习思想进行精细化调整。实验结果表明:经过精细化调整的边界回归结果较原生边界回归算法提高23.74%。

轻量化评价指标：

（1）mAP：过平均准确度均值mAP对目标检测的准确性进行评价

```
                                   表3 网络模型轻量化前后各类目标的平均精度均值

	            Pedestrain	people	bicycle	car	van	trunk	tricycle   Awing-tricycle   bus	     motor     mAP
YOLOV3_Tiny	    15.52%	15.66%	25.19%	80.21%	43.83%	25.64%	16.75%	   14.22%	    34.85%   38.96%    22.08%
YOLOV3_ReSAM	    17.28%	15.78%	28.70%	84.31%	53.77%	42.96%	30.48%	   24.38%	    62.35%   61.57%    33.15%
Tiny-YOLOV3_ReSAM   15.33%	15.58%	26.12%	84.55%	53.84%	43.61%	24.19%	   19.52%	    63.69%   44.32%    30.07%

```

（2）FLOPs：模型的推理速度取决于浮点运算器的性能，以浮点运算数大小来统计推理模型的计算量

```

             表4 轻量化前后网络结构浮点运算量及权重文件大小
	     
网络模型	               	     模型浮点运算数BFLOPs	weights文件大小(30000 batches)
YOLOV3_Tiny（原生网络模型）	       8.28                       35.4MB

YOLOV3_ReSAM（轻量化前）	         18.243                     78.1MB

Tiny-YOLOV3_ReSAM（轻量化后）	         13.613                     42.9MB

```

（3）FPS：以多帧视频流作为图像输入，统计模型检测过程中每秒处理图像的帧数即为FPS大小来衡量模型推理速度

```
        表5 轻量化前后网络模型推理延迟及FPS大小
网络模型	        	      单张图像推理延迟/s	视频流FPS
YOLOV3_Tiny（原生网络模型）	      0.064004                 17.5
 
YOLOV3_ReSAM（轻量化前）	        0.164109                  8.5

Tiny-YOLOV3_ReSAM（轻量化后）	        0.1468690                13.2

```

实验结果：
实验结果表明：对于精度方面，压缩后模型平均精度均值（mAP0.5）达到30.07%，平均精度均值损失3.08%，其中像素占比较小的微小型目标检测精度损失严重。对于运算量方面，轻量化后模型压缩比达到25.37%，减少近1/3的运算量，同时权重参数量减少45%。对于实时性方面，压缩后模型对于单张图像推理延迟与压缩前变化不大，对于视频流文件，压缩后模型推理实时性方面由原来的8.5FPS提高至13.2FPS。 

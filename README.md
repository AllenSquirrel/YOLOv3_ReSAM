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
* 在自己的代码中嵌⼊YOLO

# 上海交通大学 工程实践与科技创新Ⅱ-B
(Gongcheng shijian yu Keji Chuangxin)
## 计算机控制小车走黑线

# 说明
## 开发环境
Ubuntu 18.04 LTS: Python 3.6 (Anaconda) + OpenCV 3.4.1 (图像处理) + NumPy (矩阵计算) + Matplotlib (显示与交互)

其他？（单片机编程环境、蓝牙通信模块）

## 运行环境
Linux / Windows

## 硬件需求
- 小车
- 带有黑线的纸板
- 电脑与手机各一台

# 进度
## 环境配置
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
```
## 采集图像
在手机上购买并安装“IP摄像头”（系统：Android / IOS）  
友情链接：[shenyaocn/IP-Camera-Bridge](https://github.com/shenyaocn/IP-Camera-Bridge)

matplotlib交互模式：
```python
plt.ion()
```

每次采集图像时都运行（`address`为`http://(用户):(密码)@(地址)/video`的格式）：
```python
cap = cv.VideoCapture(address)
ret, frame = cap.read(0)
```

注意，若第一句只运行一次，采集到的视频延迟很高。

![image collection](/shot/IMG_0298.JPG)

## 检测纸板
### 为什么要检测纸板
1. 方便进一步透视变换而不需要额外的操作（手动选点），以过滤背景的干扰；
2. 当摄像头或纸板移动时，可迅速重定位，避免小车偏离路径。

### 数据集
给纸板在不同环境下拍摄几段视频，在 OpenCV 下每隔一秒采样一次，在 Matplotlib 下用鼠标选出纸板的四个角点，使用伽马校正和透视变换进行数据增强，使数据扩增为原来的10倍，用 pickle 模块保存元数据，最后获得了810张标注好的图像。

### 训练
使用预训练的 ResNet-18，最后一层输出8个数，表示4个角点的坐标，损失函数定义为4个预测点和真实点的距离之和，准确度定义为与真数值相差5像素以内的坐标值。训练300代以后的训练曲线如下：

![accuracy](/shot/acc.png)
![loss](/shot/loss.png)

### 预测
导入训练好的模型，就可以开始预测，其中一次的预测结果如下：

![predict](/shot/Figure_4.png)

## 根据初始图像生成路径
### 透视变换
[OpenCV: Geometric Image Transformations - warpPerspective()](https://docs.opencv.org/3.4.3/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87)
```python
perspective = np.array(((0,0),(0,210),(297,210),(297,0)), dtype=np.float32)
image = cv.warpPerspective(image, cv.getPerspectiveTransform(positions, perspective), (291,210))
```

下面是（手动选点后）透视变换的展示：

![before transform](/shot/Figure_1.png)
![after transform](/shot/Figure_2.png)

### 细化黑线
#### 二值化
```python
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
_, image = cv.threshold(image, threshold, max_val, cv.THRESH_BINARY)
```
#### 膨胀（形态学操作）
让黑线小幅细化，减小下一步运算量
```python
edges = cv.dilate(edges, cv.getStructuringElement(cv.MORPH_CROSS, size, kernel))
```
#### 细化
OpenCV没有提供现成函数，方法请参考[c++opencv中线条细化算法](https://www.cnblogs.com/Summerio/p/8284602.html)

### 检测直线
霍夫变换检测直线[OpenCV: Feature Detection - HoughLinesP()](https://docs.opencv.org/3.4.3/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb)

### 生成路径
对上一步得到的线段用几何算法进一步处理：
1. 合并重合线段，连接断线
2. 将线段连成折线（忽略孤立线段）

![paths](/shot/Figure_3.png)

上图中从左至右第三幅是霍夫变换的结果，图中同时出现了重合线段、断线、孤立线段三种情况，这在下一步中被一一排除。

## 接下来是什么？

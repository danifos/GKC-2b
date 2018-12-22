# 上海交通大学 工程实践与科技创新Ⅱ-B
(Gongcheng shijian yu Keji Chuangxin)
## 计算机控制小车走黑线

# 说明
## 开发环境
Ubuntu 18.04 LTS: Python 3.6 (Anaconda) + OpenCV 3.4.1 (图像处理) + NumPy (矩阵计算) + Matplotlib (显示与交互) + pySerial (蓝牙通信)

## 运行环境
Linux / Windows

## 硬件需求
- 小车（加装带有红蓝两色的外壳）
- 带有黑线的纸板
- 电脑与手机各一台

# 图像处理
## 采集图像
在手机上购买并安装“IP摄像头”（系统：Android / IOS）  
友情链接：[shenyaocn/IP-Camera-Bridge](https://github.com/shenyaocn/IP-Camera-Bridge)

matplotlib交互模式：
```python
plt.ion()
```

最开始运行：
```python
cap = cv.VideoCapture(address)
```
`address`为`http://(用户):(密码)@(地址)/video`的格式，若中途暂停了若干秒来采集图像则在运行一次。

每次采集图像时都运行：
```python
success = False
for t in range(frames_per_read):
    success, frame = cap.read(0)
    if not success:
        os._exit(0)
```
`frames_per_read`可写死，最好为5，大了会卡，小了会有延迟。但无论取值为多少，只要摄像头已连接，`success`就不会为`False`。

![image collection](/shot/IMG_0298.JPG)

## 检测纸板
### 数据集
在地砖上拍摄带有不同图案的纸板照片，总共越200张，进行标注后使用透视变换、伽马校正、高斯噪点进行数据扩增，得到了约5000张训练图像。

### 训练
使用预训练的 ResNet-18，最后一层输出8个数，表示4个角点的坐标，损失函数定义为4个预测点和真实点的距离之和，准确度定义为与真数值相差5像素以内的坐标值。训练300代以后的训练曲线如下：

![accuracy](/shot/acc.png)

可以看到，最后训练集和验证集的准确率都达到了100%，而后来实际测试时效果也同样好。

### 预测
导入训练好的模型，就可以开始预测，其中一次的预测结果如下：

![predict](/shot/Figure_4.png)

### 精定位
预测到的角点仍有偏差，此时在每个角点的邻域内使用 OpenCV 的`goodFeaturesToTrack()`函数，将误差控制在像素以内。在此之前，使用中值滤波`medianBlur()`，排除角点周围的灰尘的干扰。结果如下：

![corner](/shot/Figure_5.png)

## 根据初始图像生成路径
### 透视变换
```python
size = np.array((width, height))
perspective = np.array(((0,0), (0,height), (width,height), (width,0)),
                       dtype=np.float32)
image = cv.warpPerspective(
    image, cv.getPerspectiveTransform(positions, perspective), tuple(size)
)
```

下面是（手动选点后）透视变换的展示：

![before transform](/shot/Figure_1.png)
![after transform](/shot/Figure_2.png)

### 细化黑线
#### 聚类分析
为了使下一步的二值化对光线具有稳定性，类似于有损图像压缩，使用 k-means 聚类算法，找出纸板上黑色和白色两类颜色，并将两个聚类中心的中点作为二值化的阈值。实际上，聚类已经实现了二值化，而且允许纸板上的颜色不是黑白两色。
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
~~OpenCV没有提供现成函数，方法请参考[c++opencv中线条细化算法](https://www.cnblogs.com/Summerio/p/8284602.html)~~

细化函数没有达到预期效果，因此完全以膨胀操作代替。

### 检测直线
霍夫变换检测直线[OpenCV: Feature Detection - HoughLinesP()](https://docs.opencv.org/3.4.3/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb)

### 生成路径
对上一步得到的线段用几何算法进一步处理：
1. 合并重合线段，连接断线
2. 将线段连成折线（忽略孤立线段）

![paths](/shot/Figure_3.png)

上图中从左至右第三幅是霍夫变换的结果，图中同时出现了重合线段、断线、孤立线段三种情况，这在下一步中被一一排除。

## 检测小车

### 使用 SIFT 描述符检测
使用`cv.xfeatures2d.SIFT_create()`提取小车图像模板和采集到的图像的SIFT描述符（黄色点、蓝色点），使用`cv.BFMatcher()`暴力匹配两幅图片的描述符，用最匹配的若干描述符（32个）输入`cv.findHomography()`来生成透视变换矩阵$H$，用$H$由`cv.perspectiveTransform()`得到模板图像的四个角点到检测图像的映射（橙色点），最后以对角线（黑色线）的交点作为小车中心，小车中心到车轮处两角点连线的垂线作为小车的角度，并以此画矩形（透明白色矩形），结果如下：  
![car detection](/shot/Figure_6.png)

而在视频中检测时，一个改进是，将模板和待检测图片都转换到HSV空间，效果会更好：  
![better car tracking](/shot/tracking-better.gif)

此外，我们还尝试了多模板检测。然而，在实际测试时，由于小车并不是严格的一个平面，随着视角的变化会有所不同，因此这种方法检测常常会出错。

### 使用色块检测
之后，我们给小车加装外壳，涂上红蓝两色，在HSV空间中将两种颜色提取出来，用形态学开运算去除空穴和噪点，用`cv.moments()`找到两种颜色的重心，以此来定位小车和方向。

## 追踪纸板
使用神经网络检测纸板在速度上不够快，下面我们用检测纸板时的精定位方法来追踪纸板。

我们对纸板的每一个角点各生成一个扫描窗口，窗口的中心是纸板的角点。当纸板移动时，只要没有移出这个窗口，就可以在窗口中利用`cv.goodFeaturesToTrack()`找到一个最大响应点，作为窗口的新的中心。因此如果缓慢移动或旋转纸板，计算机也能识别出纸板的位置。

## 更换起点和终点
利用 Matplotlib 的交互功能，在图像中点击相应的点，就可以将该点更换为起点或是终点，不再赘述。

# 蓝牙通信与小车控制

## 连接蓝牙
在 Windows 上，可通过添加设备连接蓝牙，端口名为`COM6`，而在 Linux 上，则通过以下方法连接蓝牙，端口名为`/dev/rfcomm0`：
```shell
$ hciconfig hci0 sspmode 1  # 产生的错误可忽略
Can't set Simple Pairing mode on hci0: Input/output error (5)
$ hciconfig hci0 sspmode
hci0:   Type: BR/EDR  Bus: USB
BD Address: AA:BB:CC:DD:EE:FF  ACL MTU: 1021:8  SCO MTU: 64:1
Simple Pairing mode: Enabled
$ hciconfig hci0 piscan
$ sdptool add SP
00:11:22:33:44:55    小车的名字
$ rfcomm connect /dev/rfcomm0 00:11:22:33:44:55 1 &
Connected /dev/rfcomm0 to 00:11:22:33:44:55 on channel 1
Press CTRL-C for hangup
```
接下来，通过pySerial模块来使用这个端口：
```python
import pyserial
ser = serial.Serial(端口名)
```
如果是 Windows 现在应该已经成功了，而在 Linux 上，可能产生报错`[Error 16] Device or resource busy`，这时作如下操作可能可以解决：
```shell
$ sudo fuser /dev/rfcomm0
aaaa bbbb           # 正在使用此端口的进程号
$ pgrep rfcomm
cccc
dddd                # 名为rfcomm的进程号
$ sudo fuser –k /dev/rfcomm0   # 杀死 aaaa和bbbb
$ # 或者
$ sudo kill -9 aaaa bbbb cccc dddd
```

之后，通过`ser.write(str.encode(xxx))`进行通信。

## 小车逻辑
简单地，小车在前进和旋转两个状态间切换。首先小车通过旋转对准某个点，然后开始前进，直到达到了下一个点，再进行旋转。

由于采集图像与处理图像加起来大概需要0.1到0.2秒的时间（偶尔还会因为网络原因产生延迟），当计算机判断小车旋转到了正确的方向时，小车可能已经多转了一点角度。因此，我们让小车停下后，先等待一段时间，确认小车的方向无误了再前进，否则继续转向。

为了防止两点之间距离过大，导致小车在前进过程中走歪，我们在所有相邻的拐点之间再加上了它们的中点。

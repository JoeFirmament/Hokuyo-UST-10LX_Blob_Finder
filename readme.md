# UST10_TO_OSC使用指南

## 简介：

 > * 软件主要语言为python ,使用的主要模块有 hokuyolx,numpy,matplotlib,tkinter.
 > * 纯过程式语言，主要逻辑依赖tkinter window loop的after.
 > * ToDo:改为面向对象；移植为node.js

## 使用方法：

 > 1. 软件依赖config.conf文件，必须同目录下有该文件，并且文件格式，依照该要求[示例](https://gist.github.com/JoeFirmament/30fc971251f6ee42eeffca3723913f69)。
 > 2. 软件功能如下图所示
 
![Snipaste_2018-06-28_19-22-22.png](https://upload-images.jianshu.io/upload_images/1411122-086d630029199a3d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![Snipaste_2018-06-28_19-27-54.png](https://upload-images.jianshu.io/upload_images/1411122-90e35fbe89ddb24f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![Snipaste_2018-06-28_19-30-20.png](https://upload-images.jianshu.io/upload_images/1411122-6ae4c92415c3c059.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## 注意事项：

 > 1. 在探测区域没有Blob的时候，软件处于休眠状态，时间钟也会停止。 如果要判断是否运行正常，稍微遮挡雷达，图形中的时间会发生变化。
 > 2. Angual Interval，Distance Interval用来分隔Blob，极坐标模式下（PolarMode）先判断点之间的角度差超过Angual Interval ，再判断距离差超过Distance Interval，分隔Blob。笛卡尔坐标系下只用Distance Interval来分隔。
 > 3. 尺寸超过SizeThreshold值的blob，才会被识别。
 > 4. 添加的Map mode 和Raw mode 。 Map模式代表以左至右[-0.5,0.5],近处至远处[0,1]的范围发送OSC信息；Raw模式代表以距离信息发送osc信息。
 > 5. Map模式下，X轴原点为左右距离的中点。Y轴原点为Near点

 ## 相关链接：
 > [osc4py3文档](http://osc4py3.readthedocs.io/en/latest/)

 > [osc4py3@github](https://github.com/Xinne/osc4py3)

 >[hokuyolx@github](https://github.com/SkoltechRobotics/hokuyolx)

 >[hokuyolx文档](http://hokuyolx.readthedocs.io/en/latest/)
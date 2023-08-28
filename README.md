## yolov5-fire
yolov5-fire：基于YoloV5的火灾检测系统，将深度学习算法应用于火灾识别与检测领域，致力于研发准确高效的火灾识别与检测方法，实现图像中火灾区域的定位，为火灾检测技术走向实际应用提供理论和技术支持。

<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/11.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/11.png">
 <img alt="YOUR-ALT-TEXT" src="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/11.png">
</picture>

## 环境配置
基于 Windows10 操作系统，python3.7，torch1.20，cuda11以及torchvision0.40的环境，使用VOC格式数据集进行训练。  
训练前将标签文件放在fire_yolo_format文件夹下的labels文件夹中，训练前将图片文件放在fire_yolo_format文件夹下的images文件夹中。

## 训练样本集设计
<details open>
<summary>从线上收集了2059张包含起火点事物的图片，组合训练集和测试集，训练集包括1442张图像，测试集包括617张图像，通过labelimg对起火位置进行标注，如图所示。</summary>
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/1.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/1.png">
 <img alt="YOUR-ALT-TEXT" src="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/1.png">
</picture>
</details>

## 模型训练过程
<details open>
<summary>模型训练流程图、训练过程及测试结果</summary>
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/2.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/2.png">
 <img alt="YOUR-ALT-TEXT" src="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/2.png">
</picture>  

  
  模型训练  
  
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/3.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/3.png">
 <img alt="YOUR-ALT-TEXT" src="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/3.png">
</picture>  


<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/4.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/4.png">
 <img alt="YOUR-ALT-TEXT" src="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/4.png">
</picture> 

模型检测  

<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/5.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/5.png">
 <img alt="YOUR-ALT-TEXT" src="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/5.png">
</picture>
</details>

## 基于Yolov5的火灾检测系统
<details open>
<summary>系统界面设计及效果图</summary>
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/11.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/11.png">
 <img alt="YOUR-ALT-TEXT" src="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/11.png">
</picture>  
  
  图片检测界面  
  
  <picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/12.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/12.png">
 <img alt="YOUR-ALT-TEXT" src="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/12.png">
</picture>  

  摄像头实时检测界面  
  
  <picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/13.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/13.png">
 <img alt="YOUR-ALT-TEXT" src="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/13.png">
</picture>  

  视频文件检测界面  
  
    <picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/14.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/14.png">
 <img alt="YOUR-ALT-TEXT" src="https://github.com/usernameisalreadytaKeN1122/yolov5-fire/blob/main/pic/14.png">
</picture>
</details>


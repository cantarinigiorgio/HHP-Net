# HHP-Net: A light Heteroscedastic neural network for Head Pose estimation with uncertainty

![supported versions](https://img.shields.io/badge/python-3.x-brightgreen/?style=flat&logo=python&color=green)
![Library](https://img.shields.io/badge/library-Tensorflow-blue?logo=Tensorflow)
![GitHub license](https://img.shields.io/cocoapods/l/AFNetworking)

This repository contains the cource code for the paper [**HHP-Net: A light Heteroscedastic neural network for Head Pose estimation with uncertainty [WACV22]**](https://arxiv.org/abs/2111.01440).

**Code Author: Giorgio Cantarini**

Any questions or discussions are welcomed!

<p align="center">
    <img src=imgs/pipeline.png height="280"/>  
</p>

## Abstract
In this paper we introduce a novel method to estimate the head pose of people in single images starting from a small set of
head keypoints. To this purpose, we propose a regression model that exploits keypoints and outputs the head pose represented by yaw, pitch, 
and roll. Our model is simple to implement and more efficient with respect to the state of the art -- faster in inference and smaller in terms 
of memory occupancy --  with comparable accuracy.
Our method also provides a measure of the heteroscedastic uncertainties associated with the three angles, through an appropriately designed 
loss function. As an example application, we address social interaction analysis in images: we propose an algorithm for a 
quantitative estimation of the level of interaction between people, starting from their head poses and reasoning on their mutual positions.

## Installation

To clone the repository:
```bash
git clone https://github.com/cantarinigiorgio/HHP-Net
```

To install the requirements:
```bash
pip install -r requirements.txt
```

## Network architecture

<p align="center">
    <img src=imgs/network_architecture.png height="250"/>  
</p>

## Demo
<p align="center">
    <img src=imgs/points.png height="250"/> <img src=imgs/axis.png height="250"/>
</p>

There are different choices for the key points detector: in this repository we propose two variants
- a `normal` version: accurate but less efficient
- a `faster` version: less accurate but faster

### Normal version
We test three different backbones of CenterNet (HourGlass104, Resnet50V2 and Resnet50V1 available in the [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)); 
each model takes as input 512x512 images.

Download one of the previous model (e.g. [HourGlass104](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz)) then extract it to `HHP-Net/centernet/` with:
```bash
tar -zxvf centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz -C /HHP-Net/centernet
```

To make inference on a single image, run:

````
python inference_on_image.py [--detection-model PATH_DETECTION_MODEL] [--hhp-model PATH_HHPNET] [--image PATH_IMAGE]  
````
<p align="center">
    <img src=imgs/1_points.png height="250"/><img src=imgs/1_pose.png height="250"/>  
</p>

To make inference on frames from the webcam, run:

````
python inference_on_webcam.py [--detection-model PATH_DETECTION_MODEL] [--hhp-model PATH_HHPNET] 
````


### Faster version

To estimate the keypoints firstly we use an object detection model for detecting people; then we exploit a model to estimate the pose of each people detected by the previous model in the image.

In order to detect people we test [Centernet MobilenetV2](http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz): 
download it and then extract it to `HHP-Net/centernet/`:

```bash
tar -zxvf centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz -C /HHP-Net/centernet
```

Then download [Posenet](https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite) for pose estimation and move to `HHP-Net/posenet/`
```bash
mv posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite HHP-Net/posenet/
```

To make inference on a single image, run:

````
python fast_inference_on_image.py [--detection-model PATH_MODEL_DETECTION] [--pose-model PATH_MODEL_POSE] [--hhp-model PATH_HHPNET] [--image PATH_IMAGE] 
````
<p align="center">
    <img src=imgs/fast_1_points.png height="250"/><img src=imgs/fast_1_pose.png height="250"/>  
</p>

To make inference on frames from the webcam, run:

````
python fast_inference_on_webcam.py [--detection-model PATH_MODEL_DETECTION] [--pose-model PATH_MODEL_POSE] [--hhp-model PATH_HHPNET] 
````

To make inference on frames from a video, run:

````
python fast_inference_on_webcam.py [--detection-model PATH_MODEL_DETECTION] [--pose-model PATH_MODEL_POSE] [--hhp-model PATH_HHPNET] [--video PATH_VIDEO]
````


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

``` 
@misc{cantarini2021hhpnet,
      title={HHP-Net: A light Heteroscedastic neural network for Head Pose estimation with uncertainty}, 
      author={Giorgio Cantarini and Federico Figari Tomenotti and Nicoletta Noceti and Francesca Odone},
      year={2021},
      eprint={2111.01440},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Licence

MIT
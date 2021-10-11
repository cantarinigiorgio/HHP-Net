HHP-Net: A light Heteroscedastic neural network for Head Pose estimation with uncertainty
===

**Giorgio Cantarini, Francesca Odone, Nicoletta Noceti, Federico Tomenotti - WACV 2022**

**Abstract:** In this paper we introduce a novel method to estimate the head pose of people in single images starting from a small set of
head keypoints. To this purpose, we propose a regression model that exploits keypoints and outputs the head pose represented by yaw, pitch, 
and roll. Our model is simple to implement and more efficient with respect to the state of the art -- faster in inference and smaller in terms 
of memory occupancy --  with comparable accuracy.
Our method also provides a measure of the heteroscedastic uncertainties associated with the three angles, through an appropriately designed 
loss function. As an example application, we address social interaction analysis in images: we propose an algorithm for a 
quantitative estimation of the level of interaction between people, starting from their head poses and reasoning on their mutual positions.
[**ArXiv**](https://arxiv.org/)  


Any questions or discussions are welcomed!




## Installation

To download the repository:
```bash
git clone https://github.com/Malga-Vision/HHP-Net.git
```

To install the requirements:
```bash
pip install -r requirements.txt
```

## Network architecture
<img src=imgs/network_architecture.png height="250"/>  

## Demo

<img src=imgs/points.png height="250"/> <img src=imgs/axis.png height="250"/> 


There are different choices for the Keypoints detector, in this repository we test CenterNet HourGlass104 Keypoints 512x512, CenterNet Resnet50 V2 Keypoints 512x512 and CenterNet Resnet50 V1 FPN Keypoints 512x512 from the TensorFlow 2 Detection Model Zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

Download one of the previous model (e.g. http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz)

then extract it in `HHP-Net/centernet/` with:
```bash
tar -zxvf centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz -C path_to/HHP-Net/centernet
```

To make inference on a single image, run

````
python inference_on_image.py [--model-detection PATH_MODEL_DETECTION] [--image PATH_IMAGE] [--hpe-model PATH_HPPNET] 
````

To make inference on the images coming from the webcam, run

````
python inference_on_webcam.py [--model-detection PATH_MODEL_DETECTION] [--hpe-model PATH_HPPNET] 
````


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@{,
  title={},
  author={},
  journal={},
  year={}
}

```

## Code Author
- Giorgio Cantarini - Imavis s.r.l. and Malga (Machine Learning Genoa Center)

## License

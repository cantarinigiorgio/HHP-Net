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
git clone https://github.com/..
```

To install the requirements:
```bash
pip install -r requirements.txt
```


## Demo

<img src=imgs/points.png height="250"/> <img src=imgs/axis.png height="250"/> 

Download Centernet model from http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz

extract it in `HHP-Net/centernet/` with:
```bash
tar -zxvf centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz -C path_to/HHP-Net/centernet1
```

To make inference on a single image, run

````
python inference_on_image.py [--model-detection PATH_MODEL_DETECTION] [--image PATH_IMAGE] [--hpe-model PATH_HPPNET] 
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

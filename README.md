<div align="center">
  <img width="450" src="assets/Flask_logo.svg">
</div>

# Flask API for panoptic segmentation using detectron2

<div align="center">
  <a href="https://www.buymeacoffee.com/hdnh2006" target="_blank">
    <img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee">
  </a>
</div>

This code uses some utilities from [Ultralytics](https://github.com/ultralytics/ultralytics), so, thanks once again to them.
It has all the functionalities that the original code has:
- Different source: images, videos, webcam, RTSP cameras.
- Just pretrained weights is supported: COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml.

The API can be called in an interactive way, and also as a single API called from terminal.

The [model](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-panoptic-segmentation-baselines-with-panoptic-fpn) is downloaded automatically from the [detectron2](https://github.com/facebookresearch/detectron2/) repo on first use.

<img width="1024" src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png">

## Requirements

Python 3.8 or later with all [requirements.txt](requirements.txt) dependencies installed, including `torch>=1.8`. To install you will need to install `torch` and `torchvision` as it is required by detectron2 according to the installation [instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html):
```bash
$ pip3 install torch torchvision
```
And then you just need to run the requiremtns:
```bash
$ pip3 install -r requirements.txt
```

## [Panoptic](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-panoptic-segmentation-baselines-with-panoptic-fpn) segmentation

`panoptic_api.py` can deal with several sources and can run into the cpu, but it is highly recommendable to run in gpu.

```bash
Usage:
    $ python panoptic_api.py
```

## Interactive implementation implemntation

You can deploy the API able to label an interactive way.

Run:

```bash
$ python panoptic_api.py
```
Open the application in any browser 0.0.0.0:5000 and upload your image or video as is shown in video above.


## How to use the API

### Interactive way
Just open your favorite browser and go to 0.0.0.0:5000 and intuitevely load the image you want to label and press the buttom "Upload image".

The API will return the image or video labeled.

![Zidane bbox(assets/bus.jpg)


### Call from terminal or python script
The `client.py` code provides several example about how the API can be called. A very common way to do it is to call a public image from url and to get the coordinates of the bounding boxes:

```python
import requests

resp = requests.get("http://0.0.0.0:5000/predict?source=https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg&save_txt=T",
                    verify=False)
print(resp.content)

```
And you will get a json with the following data:

```
b'{"results": [{"id": 1, "isthing": true, "score": 0.9992121458053589, "category_id": 0, "instance_id": 0, "name": "person"}, {"id": 2, "isthing": true, "score": 0.9988304972648621, "category_id": 0, "instance_id": 1, "name": "person"}, {"id": 3, "isthing": true, "score": 0.9986878037452698, "category_id": 0, "instance_id": 2, "name": "person"}, {"id": 4, "isthing": true, "score": 0.9967144727706909, "category_id": 5, "instance_id": 3, "name": "bus"}, {"id": 5, "isthing": true, "score": 0.9396445155143738, "category_id": 0, "instance_id": 4, "name": "person"}, {"id": 6, "isthing": false, "category_id": 21, "area": 5421, "name": "road"}, {"id": 7, "isthing": false, "category_id": 37, "area": 39392, "name": "tree"}, {"id": 8, "isthing": false, "category_id": 44, "area": 263715, "name": "pavement"}, {"id": 9, "isthing": false, "category_id": 50, "area": 163527, "name": "building"}]}'
```

## TODO
- [ ] Mant things hehe, this is an internal project
- [ ] Integrate other options like device selection
- [ ] Add new support for other models
- [ ] Docker files


## About me and contact
 
If you want to know more about me, please visit my blog: [henrynavarro.org](https://henrynavarro.org).

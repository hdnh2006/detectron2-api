#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:58:02 2023

@author: henry
"""


from flask import Flask, render_template, Response, request
import json
import argparse
import os
import sys
from pathlib import Path
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

# Ultralytics utilities
from ultralytics.data import load_inference_source
from ultralytics.utils.checks import print_args
from ultralytics.utils import MACOS, WINDOWS
from ultralytics.utils.files import increment_path
from utils.general import update_options

# Initialize paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# Initialize Flask API
app = Flask(__name__)


def load_model(opt):
    cfg = get_cfg()
    
    # Load default model
    cfg.merge_from_file(model_zoo.get_config_file(opt.model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(opt.model)
    
    # Update parameters
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = opt.conf
    cfg.MODEL.PANOPTIC_FPN.IOU_THRESHOLDS = [opt.iou] *2
    
    model = DefaultPredictor(cfg)

    return cfg, model
    

def match_names(segments_info, dictionary):
    modified_segments = dict(results=[])
    for segment in segments_info:
        segment_copy = segment.copy()  # To ensure we don't modify the original segment in-place
        if segment['isthing']:
            # For things, use the thing_dataset_id_to_contiguous_id mapping to get the index
            segment_copy['name'] = dictionary.thing_classes[segment['category_id']]
        else:
            # For stuff, use the stuff_dataset_id_to_contiguous_id mapping to get the index
            segment_copy['name'] = dictionary.stuff_classes[segment['category_id']]
        modified_segments['results'].append(segment_copy)
    return modified_segments


def predict(opt):
    """
    Perform object detection using the YOLO model and yield results.
    
    Parameters:
    - opt (Namespace): A namespace object that contains all the options for YOLO object detection,
        including source, model path, confidence thresholds, etc.
    
    Yields:
    - JSON: If opt.save_txt is True, yields a JSON string containing the detection results.
    - bytes: If opt.save_txt is False, yields JPEG-encoded image bytes with object detection results plotted.
    """
    
    # Create paths for raw data
    raw_data = Path(opt.raw_data)
    raw_data.mkdir(parents=True, exist_ok=True)
    
    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    dataset = load_inference_source(opt.source)
    
        
    # Load model and get config file
    cfg, model = model_cfg
    metadata = model.metadata
    
    # Initi values
    vid_path, vid_writer = [None] * dataset.bs, [None] * dataset.bs
    
    for batch in dataset:
        path, im0s, vid_cap, s = batch
    
        
        # Batch size != 1 is still not supported
        im = im0s[0]
        p = Path(path[0])
        save_path = str(save_dir / p.name)
        
        # Predict
        panoptic_seg, segments_info = model(im)["panoptic_seg"]
        
        # show and return results
        if opt.save_txt:
            result = match_names(segments_info, metadata)
            result_json = json.dumps(result)
            yield result_json
        else:
            result = Visualizer(im[:, :, ::-1], metadata, scale=1)
            result = result.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
            result = result.get_image()[:, :, ::-1]
            im0 = cv2.imencode('.jpg', result)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + im0 + b'\r\n')
        
        
        n = len(im0s)
        for i in range(n):
            if opt.save: # Based on ultralytics package code
                
                # get im0 as cv2 image
                im0 = result
    
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        suffix, fourcc = ('.mp4', 'avc1') if MACOS else ('.avi', 'WMV2') if WINDOWS else ('.avi', 'MJPG')
                        save_path = str(Path(save_path).with_suffix(suffix))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer[i].write(im0)
                

        
@app.route('/')
def index():
    """
    Video streaming home page.
    """
    
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def video_feed():
    if not request.files.getlist('myfile'):
        opt.source, opt.save_txt = update_options(request)
    else:
        uploaded_file = request.files['myfile']
        source = Path(__file__).parent / raw_data / uploaded_file.filename
        uploaded_file.save(source)
        opt.save_txt = None
        opt.source = source

    return Response(predict(opt), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','--weights', type=str, default= 'COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml', help='model path or triton URL')
    parser.add_argument('--source', type=str, default= ROOT / 'data/images', help='source directory for images or videos')
    parser.add_argument('--conf','--conf-thres', type=float, default=0.25, help='object confidence threshold for detection')
    parser.add_argument('--iou', '--iou-thres', type=float, default=0.5, help='intersection over union (IoU) threshold for NMS')
    # parser.add_argument('--half', action='store_true', help='use half precision (FP16)')
    # parser.add_argument('--device', default='', help='device to run on, i.e. cuda device=0/1/2/3 or device=cpu')
    parser.add_argument('--save', action='store_true', help='save images with results')
    parser.add_argument('--exist_ok', '--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--project', default= ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--raw_data', '--raw-data', default= ROOT / 'data/raw', help='save raw images to data/raw')
    parser.add_argument('--port', default=5000, type=int, help='port deployment')
    opt, unknown = parser.parse_known_args()

    # print used arguments
    print_args(vars(opt))

    # Get port to deploy
    port = opt.port
    
    # Create path for raw data
    raw_data = Path(opt.raw_data)
    raw_data.mkdir(parents=True, exist_ok=True)
    
    # Load model (Ensemble is not supported)
    model_cfg =  load_model(opt)

    # Run app
    app.run(host='0.0.0.0', port=port, debug=False) # Don't use debug=True, model will be loaded twice (https://stackoverflow.com/questions/26958952/python-program-seems-to-be-running-twice)


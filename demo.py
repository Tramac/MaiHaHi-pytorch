# -*- coding: utf-8 -*-
"""
#####################################################################
    > File Name: demo.py
    > Author: Tramac
    > Created Time：2021/03/19 
#####################################################################
"""
import os
import sys
import yaml
import imageio
import torch
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')

from skimage.transform import resize
from skimage import img_as_ubyte
from scipy.spatial import ConvexHull

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from apps.first_order_predictor import FirstOrderPredictor


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--weight_path", default='./checkpoints/vox-cpk.pth.tar', 
                        help="path to checkpoint to restore")
    parser.add_argument("--face_detect_model", default='./checkpoints/s3fd-619a316812.pth',
                        help="path to checkpoint of S3FD")

    parser.add_argument("--source_image", default='sup-mat/source.png', 
                        help="path to source image")
    parser.add_argument("--driving_video", default='sup-mat/source.png', 
                        help="path to driving video")
    parser.add_argument("--output", default='output', help="path to output")
    parser.add_argument("--filename", default='result.mp4', help="path to output")
    parser.add_argument("--relative", dest="relative", action="store_true", 
                        help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", 
                        help="adapt movement scale based on convex hull of keypoints")
    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")
    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
                        help="Set frame to start from.")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument("--ratio", dest="ratio", type=float, default=0.4,
                        help="margin ratio")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    predictor = FirstOrderPredictor(output=args.output,
                                    filename=args.filename,
                                    weight_path=args.weight_path,
                                    face_detect_model=args.face_detect_model,
                                    config=args.config,
                                    relative=args.relative,
                                    adapt_scale=args.adapt_scale,
                                    find_best_frame=args.find_best_frame,
                                    best_frame=args.best_frame,
                                    ratio=args.ratio,
                                    cpu=args.cpu)
    predictor(args.source_image, args.driving_video)


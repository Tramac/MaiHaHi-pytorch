# -*- coding: utf-8 -*-
"""
#####################################################################
    > File Name: first_order_predictor.py
    > Author: Tramac
    > Created Time：2021/03/20 00:05:04
#####################################################################
"""
import os
import sys
import cv2
import math
import yaml
import imageio
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm
from utils.animate import normalize_kp
from skimage.transform import resize
from scipy.spatial import ConvexHull
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from sync_batchnorm import DataParallelWithCallback
from utils import face_alignment


class FirstOrderPredictor(nn.Module):
    def __init__(self, 
                 output='output', 
                 weight_path=None, 
                 face_detect_model=None,
                 config=None, 
                 relative=False,
                 adapt_scale=False, 
                 find_best_frame=False, 
                 best_frame=None, 
                 ratio=1.0,
                 cpu=False,
                 filename='result.mp4'):
        super(FirstOrderPredictor, self).__init__()
        if config is not None and isinstance(config, str):
            with open(config) as f:
                self.cfg = yaml.load(f, Loader=yaml.SafeLoader)
        elif isinstance(config, dict):
            self.cfg = config
        elif config is None:
            self.cfg = {
                'model_params': {
                    'common_params': {
                        'num_kp': 10,
                        'num_channels': 3,
                        'estimate_jacobian': True
                    },
                    'kp_detector_params': {
                        'temperature': 0.1,
                        'block_expansion': 32,
                        'max_features': 1024,
                        'scale_factor': 0.25,
                        'num_blocks': 5
                    },
                    'generator_params': {
                        'block_expansion': 64,
                        'max_features': 512,
                        'num_down_blocks': 2,
                        'num_bottleneck_blocks': 6,
                        'estimate_occlusion_map': True,
                        'dense_motion_params': {
                            'block_expansion': 64,
                            'max_features': 1024,
                            'num_blocks': 5,
                            'scale_factor': 0.25
                        }
                    }
                }
            }
            if weight_path is None:
                vox_cpk_weight_url = ""
                weight_path = ""

        self.weight_path = weight_path
        self.face_detect_model = face_detect_model
        if not os.path.exists(output):
            os.makedirs(output)
        self.output = output
        self.filename = filename
        self.relative = relative
        self.adapt_scale = adapt_scale
        self.find_best_frame = find_best_frame
        self.best_frame = best_frame
        self.ratio = ratio
        self.cpu = cpu
        self.generator, self.kp_detector = self.load_checkpoints(self.cfg, self.weight_path)

    def forward(self, source_image, driving_video):
        source_image = imageio.imread(source_image)
        bboxes = self.extract_bbox(source_image.copy())

        reader = imageio.get_reader(driving_video)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

        results = []
        for rec in bboxes:
            face_image = source_image.copy()[rec[1]:rec[3], rec[0]:rec[2]]
            face_image = resize(face_image, (256, 256))

            if self.find_best_frame or self.best_frame is not None:
                i = self.best_frame if self.best_frame is not None else self.find_best_frame_func(source_image, driving_video)
                print("Best frame: " + str(i))
                driving_forward = driving_video[i:]
                driving_backward = driving_video[:(i + 1)][::-1]
                predictions_forward = self.make_animation(
                    face_image,
                    driving_forward,
                    self.generator,
                    self.kp_detector,
                    relative=self.relative,
                    adapt_movement_scale=self.adapt_scale,
                    cpu=self.cpu)
                predictions_backward = self.make_animation(
                    face_image,
                    driving_backward,
                    self.generator,
                    self.kp_detector,
                    relative=self.relative,
                    adapt_movement_scale=self.adapt_scale,
                    cpu=self.cpu)
                predictions = predictions_backward[::-1] + predictions_forward[1:]
            else:
                predictions = self.make_animation(
                    face_image,
                    driving_video,
                    self.generator,
                    self.kp_detector,
                    relative=self.relative,
                    adapt_movement_scale=self.adapt_scale,
                    cpu=self.cpu)

            results.append({'rec': rec, 'predict': predictions})

        out_frame = []

        for i in range(len(driving_video)):
            frame = source_image.copy()
            for result in results:
                x1, y1, x2, y2 = result['rec']
                h = y2 - y1
                w = x2 - x1
                out = result['predict'][i] * 255.0
                out = cv2.resize(out.astype(np.uint8), (x2 - x1, y2 - y1))
                if len(results) == 1:
                    frame[y1:y2, x1:x2] = out
                else:
                    patch = np.zeros(frame.shape).astype('uint8')
                    patch[y1:y2, x1:x2] = out
                    mask = np.zeros(frame.shape[:2]).astype('uint8')
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cv2.circle(mask, (cx, cy), math.ceil(h * self.ratio), 
                        (255, 255, 255), -1, 8, 0)
                    frame = cv2.copyTo(patch, mask, frame)

            out_frame.append(frame)
        imageio.mimsave(
            os.path.join(self.output, self.filename), [frame for frame in out_frame], fps=fps)

    def load_checkpoints(self, config, checkpoint_path, cpu=False):
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        if not cpu:
            generator.cuda()

        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                 **config['model_params']['common_params'])
        if not cpu:
            kp_detector.cuda()

        if cpu:
            checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path)

        generator.load_state_dict(checkpoint['generator'])
        kp_detector.load_state_dict(checkpoint['kp_detector'])

        if not cpu:
            generator = DataParallelWithCallback(generator)
            kp_detector = DataParallelWithCallback(kp_detector)

        generator.eval()
        kp_detector.eval()

        return generator, kp_detector

    def make_animation(self, source_image, driving_video, generator, kp_detector,
                       relative=True, adapt_movement_scale=True, cpu=False):
        with torch.no_grad():
            predictions = []
            source = torch.tensor(source_image[np.newaxis].astype(
                np.float32)).permute(0, 3, 1, 2)

            driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(
                np.float32)).permute(0, 4, 1, 2, 3)
            kp_source = kp_detector(source)
            kp_driving_initial = kp_detector(driving[:, :, 0])

            for frame_idx in tqdm(range(driving.shape[2])):
                driving_frame = driving[:, :, frame_idx]
                if not cpu:
                    driving_frame = driving_frame.cuda()
                kp_driving = kp_detector(driving_frame)
                kp_norm = normalize_kp(
                    kp_source=kp_source,
                    kp_driving=kp_driving,
                    kp_driving_initial=kp_driving_initial,
                    use_relative_movement=relative,
                    use_relative_jacobian=relative,
                    adapt_movement_scale=adapt_movement_scale)
                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

                predictions.append(
                    np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        return predictions

    def find_best_frame_func(self, source, driving):
        def normalize_kp(kp):
            kp = kp - kp.mean(axis=0, keepdims=True)
            area = ConvexHull(kp[:, :2]).volume
            area = np.sqrt(area)
            kp[:, :2] = kp[:, :2] / area
            return kp

        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                          facedetectmodelfile=self.face_detect_model)

        kp_source = fa.get_landmarks(255 * source)[0]
        kp_source = normalize_kp(kp_source)
        norm = float('inf')
        frame_num = 0
        for i, image in tqdm(enumerate(driving)):
            kp_driving = fa.get_landmarks(255 * image)[0]
            kp_driving = normalize_kp(kp_driving)
            new_norm = (np.abs(kp_source - kp_driving)**2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i

        return frame_num

    def extract_bbox(self, image):
        detector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, flip_input=False, facedetectmodelfile=self.face_detect_model)

        frame = [image]
        predictions = detector.get_detections_for_image(np.array(frame))
        results = []
        h, w, _ = image.shape
        for rect in predictions:
            bh = rect[3] - rect[1]
            bw = rect[2] - rect[0]
            cy = rect[1] + int(bh / 2)
            cx = rect[0] + int(bw / 2)
            margin = max(bh, bw)
            y1 = max(0, cy - margin)
            x1 = max(0, cx - int(0.8 * margin))
            y2 = min(h, cy + margin)
            x2 = min(w, cx + int(0.8 * margin))
            results.append([x1, y1, x2, y2])
        boxes = np.array(results)

        return boxes


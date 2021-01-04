"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Liangchuan Gu
# DoC: 2021.01.04
# email: liangchuan.gu@robok.ai
-----------------------------------------------------------------------------------
# Description: pytorch to onnx
"""

import argparse
import sys
import os
import time

from easydict import EasyDict as edict
import cv2
import torch
import torch.onnx
import numpy as np

sys.path.append('../')

import config.kitti_config as cnf
from data_process import kitti_data_utils, kitti_bev_utils
from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder
from utils.evaluation_utils import post_processing, rescale_boxes, post_processing_v2
from utils.misc import time_synchronized
from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format


def parse_test_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
    parser.add_argument('--saved_fn', type=str, default='complexer_yolov4', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/complex_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help='the threshold for conf')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                        help='the threshold for conf')

    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_complexer_yolov4', metavar='PATH',
                        help='the video filename if the output format is video')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = '../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset', 'kitti')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.working_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs


if __name__ == '__main__':
    configs = parse_test_configs()
    configs.distributed = False  # For testing

    model = create_model(configs)
    model.print_network()
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location=torch.device('cpu')))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()

    x = torch.ones((1, 3, 608, 608)).cuda()
    torch_out = model(x)

    torch.onnx.export(model,           # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "robok_detector_3d.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
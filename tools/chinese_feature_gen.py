#!/usr/bin/env python 
# -*- coding:utf-8 -*-

# !/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.test import im_detect_feat

from layer_utils.roi_layers import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import json

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1, resnet101
from multiprocessing import Process

import torch

import pdb

CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor')

NETS = {
    'vgg16': ('vgg16_faster_rcnn_iter_%d.pth',),
    'res101': ('res101_faster_rcnn_iter_%d.pth',)
}
DATASETS = {
    'pascal_voc': ('voc_2007_trainval',),
    'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)
}

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False,
                          edgecolor='red',
                          linewidth=3.5))
        ax.text(
            bbox[0],
            bbox[1] - 2,
            '{:s} {:.3f}'.format(class_name, score),
            bbox=dict(facecolor='blue', alpha=0.5),
            fontsize=14,
            color='white')

    ax.set_title(
        ('{} detections with '
         'p({} | box) >= {:.1f}').format(class_name, class_name, thresh),
        fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(
        timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(
            torch.from_numpy(cls_boxes), torch.from_numpy(cls_scores),
            NMS_THRESH)
        dets = dets[keep.numpy(), :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Tensorflow Faster R-CNN demo')
    parser.add_argument(
        '--net',
        dest='demo_net',
        help='Network to use [vgg16 res101]',
        choices=NETS.keys(),
        default='res101')
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='Trained dataset [pascal_voc pascal_voc_0712]',
        choices=DATASETS.keys(),
        default='pascal_voc_0712')
    args = parser.parse_args()

    return args


def load_image_ids(split_name):
    ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''
    split = []
    base_dir = '/DATA/disk1/zhangming6/Datasets/AI_Challenger_2017/caption/raw_data/train_20170902'

    if split_name == 'coco_test2014':
        with open('/data/coco/annotations/image_info_test2014.json') as f:
            data = json.load(f)
            for item in data['images']:
                image_id = int(item['id'])
                filepath = os.path.join('/data/test2014/', item['file_name'])
                split.append((filepath, image_id))
    elif split_name == 'coco_test2015':
        with open('/data/coco/annotations/image_info_test2015.json') as f:
            data = json.load(f)
            for item in data['images']:
                image_id = int(item['id'])
                filepath = os.path.join('/data/test2015/', item['file_name'])
                split.append((filepath, image_id))
    elif split_name == 'genome':
        with open('/data/visualgenome/image_data.json') as f:
            for item in json.load(f):
                image_id = int(item['image_id'])
                filepath = os.path.join('/data/visualgenome/', item['url'].split('rak248/')[-1])
                split.append((filepath, image_id))
    elif split_name == 'chinese_train':
        with open(base_dir + '/caption_train_annotations_20170902.json') as f:
            for item in json.load(f):
                image_id = item['image_id']
                filepath = os.path.join(base_dir + '/caption_train_images_20170902', image_id)
                split.append((filepath, image_id))
    elif split_name == 'chinese_val':
        with open(base_dir + '/caption_validation_annotations_20170910.json') as f:
            for item in json.load(f):
                image_id = item['image_id']
                filepath = os.path.join(base_dir + '/caption_validation_images_20170910', image_id)
                split.append((filepath, image_id))
    elif split_name == 'chinese_test1':
        with open(base_dir + '/caption_test1_annotations_20170923.json') as f:
            for item in json.load(f):
                image_id = item['image_id']
                filepath = os.path.join(base_dir + '/caption_test1_images_20170923', image_id)
                split.append((filepath, image_id))
    else:
        print
        'Unknown split'
    return split


def feature_gen(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    scores, boxes, pool5 = im_detect(net, im)

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(
            torch.from_numpy(cls_boxes), torch.from_numpy(cls_scores),
            NMS_THRESH)
        dets = dets[keep.numpy(), :]
        pool5_select = pool5[keep.numpy(), :]
    # path = os.path.abspath(os.path.dirname(__file__)+'/../data/test/')
    path = 'demo_res/'
    np.save(path + 'fc.npy', pool5_select.mean(0))
    np.savez_compressed(path + 'att.npz', feat=pool5_select)
    np.save(path + 'box.npy', dets)

    print('Done!')


def feature_gen_multi(net, image_list, outpath):
    """Detect object classes in an image using pre-computed object proposals."""

    count = 0
    sum = len(image_list)
    for img_file, img_id in image_list:
        im_file = os.path.join(img_file)
        im = cv2.imread(im_file)

        scores, boxes, pool5 = im_detect(net, im)

        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(
                torch.from_numpy(cls_boxes), torch.from_numpy(cls_scores),
                NMS_THRESH)
            dets = dets[keep.numpy(), :]
            pool5_select = pool5[keep.numpy(), :]

        np.save(outpath + 'chinese_bu_fc/' + img_id + '.npy', pool5_select.mean(0))
        np.savez_compressed(outpath + 'chinese_bu_att/' + img_id + '.npz', feat=pool5_select)
        np.save(outpath + 'chinese_bu_box/' + img_id + '.npy', dets)

        count += 1
        if count % 100 == 0:
            print('{}/{}:{:.2f}%'.format(count, sum, (count / sum) * 100))

    print('Done!')


def single_img(net):
    im_names = [
        'a2af7deaa01abca741477820bbf37b340df02a88.jpg'
        # 'test_wave.jpg'
    ]
    for im_name in im_names:
        print('*' * 26)
        print('Demo for data/demo/{}'.format(im_name))
        # demo(net, im_name)
        feature_gen(net, im_name)


def multi_img(net):
    split_num = 2
    image_ids = load_image_ids('chinese_train')
    # Split image ids between gpus
    image_ids_split = [image_ids[i::split_num] for i in range(split_num)]

    procs = []
    outfile = '/DATA/disk1/zhangming6/Datasets/AI_Challenger_2017/caption/bottom_up_zm/'

    multi_process = False
    if multi_process:  # 暂不可用
        for i in range(split_num):
            p = Process(target=feature_gen_multi,
                        args=(i, net, image_ids_split[i], outfile))
            p.daemon = True
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        feature_gen_multi(net, image_ids, outfile)


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    saved_model = os.path.join(
        'output', demonet, DATASETS[dataset][0], 'default',
        NETS[demonet][0] % (70000 if dataset == 'pascal_voc' else 110000))

    if not os.path.isfile(saved_model):
        raise IOError(
            ('{:s} not found.\nDid you download the proper networks from '
             'our server and place them properly?').format(saved_model))

    # load network

    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(21, tag='default', anchor_scales=[8, 16, 32])

    net.load_state_dict(
        torch.load(saved_model, map_location=lambda storage, loc: storage))

    # net = resnet101(True)

    net.eval()
    if not torch.cuda.is_available():
        net._device = 'cpu'
    net.to(net._device)

    print('Loaded network {:s}'.format(saved_model))

    # single_img(net)
    multi_img(net)

#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from adda_model import train_adda
from fast_rcnn.train import get_training_roidb
from fast_rcnn.config import cfg,cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from networks.factory import get_network
import argparse
import pprint
import numpy as np
import sys
import pdb
import tensorflow as tf
import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
from networks.factory import get_network
import argparse
import pprint
import time, os, sys
import tensorflow as tf


#Command to Run
#python train_youtube_adda.py --device gpu --device_id 0 --weights /home/charan/target_rcnn/VGGnet_fast_rcnn_iter_1000.ckpt --imdb yto_trainval --iters 1 --cfg /home/charan/oldFaster-RCNN_TF/experiments/cfgs/faster_rcnn_end2end.yml --network VGGnet_train_target


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ADDA video network')
    parser.add_argument('--device', dest='device', help='device to use -> gpu/cpu',
                        default='gpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=70000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    device_name = '/{}:{:d}'.format(args.device,args.device_id)
    print device_name

    if args.device == 'gpu':
        cfg.USE_GPU_NMS = True
        cfg.GPU_ID = args.device_id
    else:
        cfg.USE_GPU_NMS = False

    network = get_network(args.network_name)
    print 'Use network `{:s}` in training'.format(args.network_name)

    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='target')

    # start a session
    saver = tf.train.Saver(var_list=target_vars,write_version=tf.train.SaverDef.V1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options))
    saver.restore(sess, args.pretrained_model)



    train_adda(sess,saver,target_vars,network,imdb, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)

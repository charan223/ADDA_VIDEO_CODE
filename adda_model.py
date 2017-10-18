# --------------------------------------------------------
# Copyright (c) 2017 IIT KGP
# Licensed under The IIT KGP License [dont see LICENSE for details]
# Written by Charan Reddy
# --------------------------------------------------------

"""Train a ADDA video network."""
#Faster RCNN imports
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.train import get_data_layer,filter_roidb
import gt_data_layer.roidb as gdl_roidb
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
from networks.factory import get_network

#Other imports
import numpy as np
import time, os, sys
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.contrib import slim
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.contrib.learn.python.learn.dataframe.queues.feeding_queue_runner import FeedingQueueRunner
from collections import OrderedDict,deque
import argparse
import pprint
import logging
import random
import click
from tqdm import tqdm
import pickle
from contextlib2 import ExitStack
import tflearn
import logging.config
import os.path
import yaml




#removes scope of the key to be stored in dict
def remove_first_scope(name):
    return '/'.join(name.split('/')[1:])

#collects adversary variables and returns a dictionary 
def collect_vars(scope, start=None, end=None):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    var_dict = OrderedDict()
    for var in vars[start:end]:
        var_name = remove_first_scope(var.op.name)
        var_dict[var_name] = var
    return var_dict

#returns dictionary of target variables
def collect_vars_new(vars, start=None, end=None):
    var_dict = OrderedDict()
    for var in vars[start:end]:
        var_name = remove_first_scope(var.op.name)
        var_dict[var_name] = var
    return var_dict

#adversarial discriminator
def adversarial_discriminator(net, layers, scope='adversary', leaky=False, reuse = False):
    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        if leaky:
            activation_fn = tflearn.activations.leaky_relu
        else:
            activation_fn = tf.nn.relu
        with ExitStack() as stack:
            stack.enter_context(tf.variable_scope(scope))
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=activation_fn,
                    weights_regularizer=slim.l2_regularizer(2.5e-5)))
            for dim in layers:
                net = slim.fully_connected(net, dim)
            net = slim.fully_connected(net, 2, activation_fn=None)
    return net

#global variables for custom_source_next_batch()
source_index=0
source_vectors=np.load('/home/charan/oldFaster-RCNN_TF/features/new_source_features.npy')
def custom_source_next_batch():
	global source_index
	global source_vectors
	source_datasize = 1369665
	source_batchsize = 128
	random_source_vectors=source_vectors
	if source_index % 40==0:
		idxs = np.random.permutation(source_datasize) #shuffled ordering
		random_source_vectors = source_vectors[idxs]
	i=source_index % 40
	source_batch = random_source_vectors[i * source_batchsize: (i+1) * source_batchsize]
	source_index=source_index+1
	return source_batch



class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, target_vars, saver, network, imdb, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model
	self.target_vars= target_vars
        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print 'done'

        # For checkpoint
        self.saver = saver

    def train_model(self, sess, saver, target_vars, max_iters):
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)

	#inputs
	source_ft = tf.placeholder(tf.float32, shape=(128, 4096))
        target_features = self.net.get_output('target/fc7')
        target_ft = target_features
	print('CheckPoint1: Source and Target features formed!')


        # adversarial network
	source_ft = tf.reshape(source_ft, [-1, int(source_ft.get_shape()[-1])])
	target_ft = tf.reshape(target_ft, [-1, int(target_ft.get_shape()[-1])])
	adversary_ft = tf.concat([source_ft, target_ft], 0)
	#source_adversary_label = tf.zeros([tf.shape(source_ft)[0]], tf.int32)
	#target_adversary_label = tf.ones([tf.shape(target_ft)[0]], tf.int32)
	#adversary_label = tf.concat([source_adversary_label, target_adversary_label], 0)
	#adversary_logits = adversarial_discriminator(adversary_ft, [500,500], leaky=True)
	
        D_logits = adversarial_discriminator(source_ft, [500, 500], leaky=True, reuse=False)  # source logits (real)
        D_logits_ = adversarial_discriminator(target_ft, [500, 500], leaky=True, reuse=True)  # target logits (fake)

        # discriminator losses

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D_logits)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_logits_)))
        d_loss = d_loss_real + d_loss_fake

        # generator losses: in our case target RCNN losses

        target_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_logits_)))

        print('CheckPoint2: Adversarial Network Formed!')

	# losses
	#mapping_loss = tf.losses.sparse_softmax_cross_entropy(1 - adversary_label, adversary_logits)
	#adversary_loss = tf.losses.sparse_softmax_cross_entropy(adversary_label, adversary_logits)	
	print('CheckPoint3: Loss Operations Formed!')

	# variable collection
	target_vars = collect_vars_new(target_vars)
	adversary_vars = collect_vars('adversary')
	print('CheckPoint4: Variable Collection Done!')

	# optimizer
	lr = 0.0002
	lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
	optimizer = tf.train.AdamOptimizer(lr_var, 0.5)
	discriminator_step = optimizer.minimize(d_loss, var_list=list(adversary_vars.values()))
       	target_step = optimizer.minimize(target_loss, var_list=list(target_vars.values()))

 	print('CheckPoint5: Optimizer Declaration Done!')


	# optimization loop (finally)
	output_dir = os.path.join('snapshot/')
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	target_losses = deque(maxlen=10)
	disc_losses = deque(maxlen=10)

        # iintialize variables
        sess.run(tf.global_variables_initializer())
        timer = Timer()
	snapping_iters = 500
	display_iters = 1
        for i in range(max_iters):

            # get one batch
            blobs = data_layer.forward()
            x=custom_source_next_batch()
	    feed_dict={source_ft: x, self.net.data: blobs['data'], self.net.im_info: blobs['im_info'], self.net.keep_prob: 0.5, \
                           self.net.gt_boxes: blobs['gt_boxes']}
            print("will start training::")
            # Update the discriminator
            disc_loss_val, _ = sess.run([d_loss, discriminator_step], feed_dict = feed_dict)
            print("Disc trained")
            # Update the target generator 
            target_loss_val, _ = sess.run([target_loss, target_step], feed_dict = feed_dict)

	    target_losses.append(target_loss_val)
	    disc_losses.append(disc_loss_val)
	    f=open('losses.txt','a')
	    if i % display_iters == 0:
			print >>f, ('{:} Target: {:10.4f}     (avg: {:10.4f})'
                        '    Disc: {:10.4f}     (avg: {:10.4f})'
                        .format('Iteration {}:'.format(i),
                                target_loss_val,
                                np.mean(target_losses),
                                disc_loss_val,
                                np.mean(disc_losses)))
	    if (i + 1) % snapping_iters == 0:
			snapshot_path = saver.save(sess, os.path.join(output_dir, 'yto'), global_step=i + 1)
			print('Saved snapshot to {}'.format(snapshot_path))
	    f.close()
	coord.request_stop()
	coord.join(threads)
	sess.close()



def train_adda(sess,saver,target_vars,network,imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a ADDA video network."""
    roidb = filter_roidb(roidb)
    sw = SolverWrapper(sess,target_vars,saver,network,imdb, roidb, output_dir, pretrained_model=pretrained_model)
    print 'Solving...'
    sw.train_model(sess,saver,target_vars, max_iters)
    print 'Done Solving'

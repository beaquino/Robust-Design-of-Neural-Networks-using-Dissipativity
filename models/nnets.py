#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:38:37 2018
by Arash Rahnama
Modified and updated on Nov 24 2021
by Bernardo Aquino
"""
import tensorflow as tf
from utility import layers as l

NUM_CLASSES = 10

def forward_net(x, num_classes=NUM_CLASSES, weight_decay=0, l2_norm=False, update_collection=None, 
                batch_normal=False, spectral_norm=True, rho_list=[1.0,1.0,1.0], reuse=None,
                training=False):

    assert len(rho_list) == 3

    lin = l.linear(x, 50, scope_name='linear1', weight_decay=weight_decay, l2_norm=l2_norm, 
                   spectral_norm=spectral_norm, update_collection=update_collection, 
                   rho=rho_list[0], reuse=reuse)
    lin1 = tf.nn.leaky_relu(lin, alpha=0.1, name='linear1_leaky_relu')

    lin = l.linear(lin1, 20, scope_name='linear2', weight_decay=weight_decay, l2_norm=l2_norm, 
                   spectral_norm=spectral_norm, update_collection=update_collection, 
                   rho=rho_list[1], reuse=reuse)
    lin2 = tf.nn.leaky_relu(lin, alpha=0.1, name='linear2_leaky_relu')

    output_layer = l.linear(lin2, num_classes, scope_name='output_layer', 
                            weight_decay=weight_decay, l2_norm=l2_norm, 
                            spectral_norm=spectral_norm, update_collection=update_collection,
                            rho=rho_list[2], reuse=reuse)
        
    return output_layer

def alexnet(x, num_classes=NUM_CLASSES, update_collection=None, spectral_norm=True,
            rho_list=[1.0,1.0,1,0,1.0,1.0], batch_normal=False, reuse=None, 
            training=False, l2_norm=False, weight_decay=0):
    assert len(rho_list) == 5

    conv = l.conv2d(x, [5, 5, 3, 96], scope_name='conv1', update_collection=update_collection, 
                    batch_normal=batch_normal, spectral_norm=spectral_norm, l2_norm=l2_norm,
                    weight_decay=weight_decay, rho=rho_list[0], reuse=reuse)
    conv1 = tf.nn.leaky_relu(conv, alpha=0.1, name='conv1_leaky_relu')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

    conv = l.conv2d(norm1, [5, 5, 96, 256], scope_name='conv2', update_collection=update_collection, 
                    batch_normal=batch_normal, spectral_norm=spectral_norm, l2_norm=l2_norm,
                    weight_decay=weight_decay, rho=rho_list[1], reuse=reuse)
    conv2 = tf.nn.leaky_relu(conv, alpha=0.1, name='conv2_leaky_relu')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
        
    x_reshape = tf.reshape(norm2, [-1, 6*6*256])
    
    lin = l.linear(x_reshape, 384, scope_name='linear1', weight_decay=weight_decay, l2_norm=l2_norm, 
                   spectral_norm=spectral_norm, update_collection=update_collection, 
                   rho=rho_list[2], reuse=reuse)
    lin1 = tf.nn.leaky_relu(lin, alpha=0.1, name='linear1_leaky_relu')
    
    lin = l.linear(lin1, 192, scope_name='linear2', weight_decay=weight_decay, l2_norm=l2_norm, 
                   spectral_norm=spectral_norm, update_collection=update_collection, 
                   rho=rho_list[3], reuse=reuse)
    lin2 = tf.nn.leaky_relu(lin, alpha=0.1, name='linear2_leaky_relu')
    
    output_layer = l.linear(lin2, num_classes, scope_name='output_layer', weight_decay=weight_decay, 
                            l2_norm=l2_norm, spectral_norm=spectral_norm, 
                            update_collection=update_collection, rho=rho_list[4], reuse=reuse)
        
    return output_layer

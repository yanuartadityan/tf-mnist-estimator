from __future__ import (absolute_import, division, print_function)
import numpy as np
import tensorflow as tf


'''CNN layer
    1. CONV layer 32-5x5 (32 5x5 pixel subregions)
    2. POOL layer 2-2x2 (2 stride 2x2)
    3. CONV layer 64-5x5 (64 5x5 pixel subregions)
    4. POOL layer 2-2x2 (2 stride 2x2)
    5. DENSE layer 16x16x12
    6. FC layer (logit) [1x1x10]
'''
def cnn_model_fn(features, labels, mode):
    """Model function for CNN"""

    # input layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # layer 1 - CONV layer 32-5x5
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    # layer 1 - POOLING
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # layer 2 - CONV layer 64-5x5
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    # layer 2 - POOLING
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # layer 3 - DENSE layer
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode==tf.estimator.ModeKeys.TRAIN
    )

    # layer 4 - LOGIT layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    # PREDICTIONS
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # LOSS
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # TRAINING OP (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # EVAL METRICS
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

import tensorflow as tf
import numpy as np


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(name + '/mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar(name + '/sttdev', stddev)
        tf.summary.scalar(name + '/max', tf.reduce_max(var))
        tf.summary.scalar(name + 'min/', tf.reduce_min(var))
        tf.summary.histogram(name, var)


def fc(tensor, output_dim, is_training_mode, name, kp_dropout, act=tf.nn.relu):
    with tf.name_scope(name):
        input_dim = tensor.get_shape()[1].value
        w_init = tf.truncated_normal([input_dim, output_dim], stddev=np.sqrt(2.0 / input_dim))
        w = tf.Variable(w_init)
        print(name, 'input  ', tensor)
        print(name, 'w  ', w.get_shape())
        variable_summaries(w, name + '/w')
        b_init = tf.constant(0.0, shape=[output_dim])
        b = tf.Variable(b_init)
        variable_summaries(b, name + '/b')
        tensor = tf.matmul(tensor, w) + b
        tensor = act(tensor)
        if kp_dropout != 1.0:
            tensor = tf.cond(is_training_mode, lambda: tf.nn.dropout(tensor, kp_dropout), lambda: tf.identity(tensor))
    return tensor


def conv(tensor, out_dim, filter_size, stride, is_training_mode, name, kp_dropout, act=tf.nn.relu):
    with tf.name_scope(name):
        in_dim_h = tensor.get_shape()[1].value
        in_dim_w = tensor.get_shape()[2].value
        in_dim_d = tensor.get_shape()[3].value
        w_init = tf.truncated_normal([filter_size, filter_size, in_dim_d, out_dim],
                                     stddev=np.sqrt(2.0 / (in_dim_h * in_dim_w * in_dim_d)))
        w = tf.Variable(w_init)
        print(name, 'input  ', tensor)
        print(name, 'w  ', w.get_shape())
        variable_summaries(w, name + '/w')
        b_init = tf.constant(0.0, shape=[out_dim])
        b = tf.Variable(b_init)
        variable_summaries(b, name + '/b')
        tensor = tf.nn.conv2d(tensor, w, strides=[1, stride, stride, 1], padding='SAME') + b
        tf.summary.image(name + '_Filtre1', tensor[:, :, :, 0:1], 5)
        tensor = act(tensor)
        if kp_dropout != 1.0:
            tensor = tf.cond(is_training_mode, lambda: tf.nn.dropout(tensor, kp_dropout), lambda: tf.identity(tensor))
    return tensor


def maxpool(tensor, pool_size, name):
    with tf.name_scope(name):
        tensor = tf.nn.max_pool(tensor,
                                ksize=(1, pool_size, pool_size, 1),
                                strides=(1, pool_size, pool_size, 1),
                                padding='SAME')
    return tensor


def flat(tensor):
    in_dim_h = tensor.get_shape()[1].value
    in_dim_w = tensor.get_shape()[2].value
    in_dim_d = tensor.get_shape()[3].value
    tensor = tf.reshape(tensor, [-1, in_dim_h * in_dim_w * in_dim_d])
    print('flat output  ', tensor)
    return tensor


def unflat(tensor, out_dim_h, out_dim_w, out_dim_d):
    tensor = tf.reshape(tensor, [-1, out_dim_h, out_dim_w, out_dim_d])
    tf.summary.image('input', tensor, 5)
    print('unflat output  ', tensor)
    return tensor

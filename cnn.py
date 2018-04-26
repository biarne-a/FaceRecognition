import tensorflow as tf
import layers


def get_dict(database, x, itm, y_desired=None):
    if y_desired is not None:
        xs, ys = database.next_training_batch()
        return {x: xs, y_desired: ys, itm: False}
    xs = database.next_training_batch(with_y=False)
    return {x: xs, itm: False}


def get_last_dict(database, x, itm):
    xs = database.last_training_batch()
    return {x: xs, itm: False}


def get_inputs_test(database):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, database.image_dim], name='x')
        itm = tf.placeholder("bool", name='Is_Training_Mode')
        return x, itm


def get_inputs_train(database):
    with tf.name_scope('input'):
        x, itm = get_inputs_test(database)
        y_desired = tf.placeholder(tf.float32, [None, 2], name='y_desired')
        return x, y_desired, itm


def get_cnn(x, itm, keep_prob_dropout):
    nb_conv_per_block = 10
    nb_filter = 16
    with tf.name_scope('CNN'):
        t = layers.unflat(x, 56, 56, 3)
        for k in range(4):
            for i in range(nb_conv_per_block):
                d = layers.conv(t, nb_filter, 3, 1, itm, 'conv33_%d_%d' % (k, i), keep_prob_dropout)
                t = tf.concat([t, d], axis=3)
            t = layers.maxpool(t, 2, 'pool')
            t = layers.conv(t, 32, 1, 1, itm, 'conv11_%d' % (k), keep_prob_dropout)
        t = layers.flat(t)
        t = layers.fc(t, 50, itm, 'fc_1', keep_prob_dropout)
        y = layers.fc(t, 2, itm, 'fc_2', kp_dropout=1.0, act=tf.nn.softmax)
        return y


def get_count_equal(t, value):
    mask = tf.equal(t, value)
    return tf.reduce_sum(tf.cast(mask, tf.int32))


def get_accuracy(y, y_desired):
    with tf.name_scope('accuracy'):
        real_y = tf.argmax(y_desired, 1)
        prediction = tf.argmax(y, 1)
        nb_real_0 = get_count_equal(real_y, 0)
        nb_real_1 = get_count_equal(real_y, 1)
        nb_pred_0 = get_count_equal(prediction, 0)
        nb_pred_1 = get_count_equal(prediction, 1)
        correct_zeros = tf.to_float(tf.reduce_sum((1 - real_y) * (1 - prediction)))
        correct_ones = tf.to_float(tf.reduce_sum(real_y * prediction))
        accuracy = 0.5 * (tf.to_float(1 / nb_real_0) * correct_zeros + tf.to_float(1 / nb_real_1) * correct_ones)
        return nb_real_0, nb_real_1, nb_pred_0, nb_pred_1, correct_zeros, correct_ones, accuracy


def get_cross_entropy(y, y_desired):
    with tf.name_scope('cross_entropy'):
        diff = y_desired * tf.log(y)
        with tf.name_scope('total'):
            cross_entropy = -tf.reduce_mean(diff)
        tf.summary.scalar('cross entropy', cross_entropy)
        return cross_entropy


def get_learning_rate():
    with tf.name_scope('learning_rate'):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(1e-3, global_step, 1000, 0.75, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        return global_step, learning_rate


def get_output(cnn):
    return tf.argmax(cnn, 1)

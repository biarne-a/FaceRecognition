import tensorflow as tf
import datasets as ds
import cnn
import signal
import sys


def save_before_exit(saver):
    print("Saving model before exit")
    saver.save(sess, "models/interrupted_model.ckpt")
    sys.exit(0)


load_model = False
keep_prob_dropout = 1.0

experiment_name = '10k_Dr%.3f' % keep_prob_dropout
train = ds.DataSet('Data/db_train_80.raw', nb_data=int(111430 * 0.8), filename_labels='Data/label_train_80.txt',
                   batch_size=256, balance=True)
test = ds.DataSet('Data/db_test_20.raw', nb_data=int(111430 * 0.2), filename_labels='Data/label_test_20.txt')


x, y_desired, itm = cnn.get_inputs_train(train)
y = cnn.get_cnn(x, itm, keep_prob_dropout)
output = cnn.get_output(y)
cross_entropy = cnn.get_cross_entropy(y, y_desired)

nb_real_0, nb_real_1, nb_pred_0, nb_pred_1, correct_zeros, correct_ones, accuracy = \
    cnn.get_accuracy(y, y_desired)

global_step, learning_rate = cnn.get_learning_rate()
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
merged = tf.summary.merge_all()

acc_test = tf.placeholder("float", name='Acc_Test')
accuracy_test = tf.summary.scalar('Acc_Test', acc_test)


print("-----------------------------------------------------")
print("-----------", experiment_name)
print("-----------------------------------------------------")


sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
writer = tf.summary.FileWriter(experiment_name, sess.graph)
saver = tf.train.Saver(max_to_keep=None)
if load_model:
    saver.restore(sess, "./model.ckpt")

model_saver = ModelSaver(sess, saver)

signal.signal(signal.SIGINT, lambda s, f: save_before_exit(saver))

nb_it = 5000
for it in range(nb_it):
    dict = cnn.get_dict(train, x, itm, y_desired=y_desired)
    sess.run(train_step, feed_dict=dict)

    if it % 10 == 0:
        ce, lr = sess.run([cross_entropy, learning_rate], feed_dict=dict)
        # pred_y = sess.run([y], feed_dict=dict)
        acc_tensors = [nb_pred_0, nb_pred_1, correct_zeros, correct_ones, accuracy]
        nb_pred_zs, nb_pred_os, cor_zs, cor_os, acc = sess.run(acc_tensors, feed_dict=dict)
        print("it= %6d - rate= %f - cross_entropy= %f" % (it, lr, ce))
        # print("pred_y = " + str(pred_y))
        print("nb pred zeros= %f - nb pred ones= %f - correct zeros= %f - correct ones = %f - acc= %f" %
              (nb_pred_zs, nb_pred_os, cor_zs, cor_os, acc))

    if it % 50:
        summary_merged = sess.run(merged, feed_dict=dict)
        writer.add_summary(summary_merged, it)

    if it % 200 == 0 and it > 0:
        acc_test_value = test.mean_accuracy(sess, accuracy, x, y_desired, itm)
        print("mean accuracy test = %f" % (acc_test_value))
        summary_acc = sess.run(accuracy_test, feed_dict={acc_test: acc_test_value})
        writer.add_summary(summary_acc, it)
        saver.save(sess, "models/intermediate_%d.ckpt" % it)



writer.close()
saver.save(sess, "models/final_model.ckpt")
sess.close()

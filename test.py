import tensorflow as tf
import numpy as np
import cnn
import datasets as ds
from model_saver import Range
from model_saver import ModelSaver


keep_prob_dropout = 1.0
test_mode = False


def print_percents(y):
    zeros = y[y == 0]
    ones = y[y == 1]
    print("nb_0_percentage %f" % (zeros.shape[0] / y.shape[0]))
    print("nb_1_percentage %f" % (ones.shape[0] / y.shape[0]))


def flatten_ys(all_y):
    for y in all_y:
        for y_i in y:
            yield y_i


def get_prediction(sess, net, db, x, itm):
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("test", sess.graph)
    nb_rounds = int(db.nb_data / db.batch_size)
    all_y = []
    for i in range(nb_rounds):
        dict = cnn.get_dict(db, x, itm)
        pred = sess.run(net, feed_dict=dict)
        y = sess.run(tf.argmax(pred, 1), feed_dict=dict)
        all_y.append(y)
        summary_merged = sess.run(merged, feed_dict=dict)
        writer.add_summary(summary_merged, 0)
    if db.nb_data % db.batch_size > 0:
        dict = cnn.get_last_dict(db, x, itm)
        pred = sess.run(net, feed_dict=dict)
        y = sess.run(tf.argmax(pred, 1), feed_dict=dict)
        all_y.append(y)
        summary_merged = sess.run(merged, feed_dict=dict)
        writer.add_summary(summary_merged, 0)

    writer.close()

    return np.array(list(flatten_ys(all_y)))


def run_prediction(sess, db, x, itm, output):
    y = get_prediction(sess, output, db, x, itm)
    np.savetxt("val_pred.txt", y, fmt="%d")


def print_accuracy(sess, db, x, y_desired, itm, accuracy):
    all_accuracies = []
    for i in range(20):
        acc = sess.run(accuracy, feed_dict=cnn.get_dict(db, x, itm, y_desired))
        all_accuracies.append(acc)
        print("acc = %f" % acc)
    mean_acc = np.mean(all_accuracies)
    print("mean acc = %f" % mean_acc)


def predict_for_model(model_name, pred_name, net, db, x, itm):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_name)

        y = get_prediction(sess, net, db, x, itm)
        print("len y = %d" % len(y))
        np.savetxt(pred_name, y, fmt="%d")
        print("Prediction for %s" % model_name)
        print_percents(y)
        print("")


# def predict_saved_models():
#     saver = ModelSaver()
#     for range in saver.ranges_:
#         for nb in range(Range.MAX_SAVE):
#             model_name = range.get_model_name()
#             pred_name = "val_pred_%0.3f-%0.3f_%d.txt" % (range.low_bound_, range.high_bound_, nb)
#             predict_for_model(model_name, pred_name, output, db, x, itm)


if __name__ == "__main__":
    db = ds.DataSet('Data/db_val.raw', 10130, batch_size=100, predict=True)
    x, itm = cnn.get_inputs_test(db)
    net = cnn.get_cnn(x, itm, keep_prob_dropout)
    # output = cnn.get_output(net)
    predict_for_model("models/intermediate_3000.ckpt", "val_pred.txt", net, db, x, itm)

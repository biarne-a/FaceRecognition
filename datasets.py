import numpy as np


class DataSet(object):
    def __init__(self, filename_data, nb_data, l2_normalize=False, batch_size=128, filename_labels=None,
                 balance=False, predict=False):
        # ---- Attributes ----
        self.nb_data = nb_data
        # image size
        self.image_dim = 56 * 56 * 3
        self.data = None
        self.labels = None
        self.x = None
        self.y_desired = None
        self.batch_size = batch_size
        self.half_batch_size = int(batch_size / 2)
        self.cur_pos = 0
        self.balance = balance

        print("loading data")
        with open(filename_data, 'rb') as f:
            self.data = np.fromfile(f, dtype=np.uint8, count=nb_data * self.image_dim).astype(np.float32)
            self.data = self.data.reshape(nb_data, self.image_dim)

        if filename_labels is not None:
            self.labels = np.loadtxt(filename_labels, dtype=np.float64)
            self.nb_labels_1 = np.sum(self.labels)
            self.nb_labels_0 = self.labels.shape[0] - self.nb_labels_1
            print('nb ones : ', self.nb_labels_1)
            print('nb zeros : ', self.nb_labels_0)
            half_zeros = np.array([0] * self.half_batch_size)
            half_ones = np.array([1] * self.half_batch_size)
            half_half_labels = np.hstack((half_zeros, half_ones)).reshape(-1, 1)
            self.half_half_labels = self.prepare_labels(half_half_labels)

        print('nb data : ', self.nb_data)

        if not predict:
            self.data, self.labels = self.shuffle_and_sort(self.data, self.labels)

        if filename_labels is not None:
            self.zero_indexes = np.argwhere(self.labels == 0).reshape(1, -1)[0]
            self.one_indexes = np.argwhere(self.labels == 1).reshape(1, -1)[0]
            self.labels = self.prepare_labels(self.labels)
        
        if l2_normalize:
            self.data /= np.sqrt(np.expand_dims(np.square(self.data).sum(axis=1), 1))

    def prepare_labels(self, labels):
        return np.array([np.array([0, 1]) if bool(l) else np.array([1, 0]) for l in labels])

    def shuffle_and_sort(self, data, labels=None):
        rng_state = np.random.get_state()
        np.random.shuffle(data)
        if labels is not None:
            np.random.set_state(rng_state)
            np.random.shuffle(labels)
        return data, labels

    def pick_indexes(self, from_indexes):
        batch_indexes = np.random.choice(from_indexes, self.half_batch_size)
        return self.data[batch_indexes]

    def next_balanced_training_batch(self):
        batch_zeros = self.pick_indexes(self.zero_indexes)
        batch_ones = self.pick_indexes(self.one_indexes)
        data = np.vstack((batch_zeros, batch_ones))
        return self.shuffle_and_sort(data.copy(), self.half_half_labels.copy())

    def next_unbalanced_training_batch(self, with_y=True):
        if self.cur_pos + self.batch_size > self.nb_data:
            self.cur_pos = 0
        xs = self.data[self.cur_pos:self.cur_pos + self.batch_size, :]
        if with_y:
            ys = self.labels[self.cur_pos:self.cur_pos + self.batch_size, :]
            res = xs, ys
        else:
            res = xs
        self.cur_pos += self.batch_size
        return res

    def next_training_batch(self, with_y=True):
        if self.balance:
            return self.next_balanced_training_batch()
        return self.next_unbalanced_training_batch(with_y)

    def last_training_batch(self):
        xs = self.data[self.cur_pos:, :]
        self.cur_pos = 0
        return xs

    def mean_accuracy(self, tf_session, loc_acc, loc_x, loc_y, loc_is_train):
        acc = 0
        for i in range(0, self.nb_data, self.batch_size):
            cur_batch_size = min(self.batch_size, self.nb_data - i)
            feed_dict = {loc_x: self.data[i:i + cur_batch_size, :],
                         loc_y: self.labels[i:i + cur_batch_size, :],
                         loc_is_train: False}
            acc += tf_session.run(loc_acc, feed_dict) * cur_batch_size
        acc /= self.nb_data
        return acc

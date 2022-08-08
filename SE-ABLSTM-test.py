from sklearn.metrics import accuracy_score,confusion_matrix
from keras.models import load_model, Sequential
from numpy import std, mean, array, argmax
from numpy import mean
from numpy import std
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import glob
import os
import csv
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def merge_csi_label(csifile, labelfile, win_len=200, thrshd=0.6, step=50):
    """
    Merge CSV files into a Numperrory Array  X,  csi amplitude feature
    Returns Numpy Array X, Shape(Num, Win_Len, 64)
    Args:
        csifile  :  str, csv file containing CSI data
        labelfile:  str, csv fiel with activity label
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    activity = []
    with open(labelfile, 'r') as labelf:
        reader = csv.reader(labelf)
        for line in reader:
            label = line[0]
            if label == 'NoActivity':
                activity.append(0)
            else:
                activity.append(1)
    activity = np.array(activity)
    csi = []
    with open(csifile, 'r') as csif:
        reader = csv.reader(csif)
        for line in reader:
            line_array = np.array([float(v) for v in line])
            # extract the amplitude only
            line_array = line_array[1:65]
            csi.append(line_array[np.newaxis, ...])
    csi = np.concatenate(csi, axis=0)
    assert (csi.shape[0] == activity.shape[0])
    # screen the data with a window
    index = 0
    feature = []
    while index + win_len <= csi.shape[0]:
        cur_activity = activity[index:index + win_len]
        if np.sum(cur_activity) < thrshd * win_len:
            index += step
            continue
        cur_feature = np.zeros((1, win_len, 64))
        cur_feature[0] = csi[index:index + win_len, :]
        feature.append(cur_feature)
        index += step
    return np.concatenate(feature, axis=0)


def extract_csi_by_label(raw_folder, label, labels, save=False, win_len=200, thrshd=0.6, step=50):
    """
    Returns all the samples (X,y) of "label" in the entire dataset
    Args:
        raw_folder: The path of Dataset folder
        label    : str, could be one of labels
        labels   : list of str, ['wave', 'clap', 'walk', 'liedown', 'sitdown', 'fall', 'pickup']
        save     : boolean, choose whether save the numpy array
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    print('Starting Extract CSI for Label {}'.format(label))
    label = label.lower()
    if not label in labels:
        raise ValueError(
            "The label {} should be among 'wave', 'clap', 'walk', 'liedown', 'sitdown', 'fall', 'pickup'".format(labels))

    data_path_pattern = os.path.join(raw_folder, label, 'user_*' + label + '*.csv')
    input_csv_files = sorted(glob.glob(data_path_pattern))
    # annot_csv_files = [os.path.basename(fname).replace('user_', 'annotation_user') for fname in input_csv_files]
    # annot_csv_files = [os.path.join(raw_folder, label, fname) for fname in annot_csv_files]
    annot_csv_files = os.path.join(raw_folder, label, 'Annotation_user_*' + label + '*.csv')
    annot_csv_files = sorted(glob.glob(annot_csv_files))
    feature = []
    index = 0
    for csi_file, label_file in zip(input_csv_files, annot_csv_files):
        index += 1
        if not os.path.exists(label_file):
            print('Warning! Label File {} doesn\'t exist.'.format(label_file))
            continue
        feature.append(merge_csi_label(csi_file, label_file, win_len=win_len, thrshd=thrshd, step=step))
        print('Finished {:.2f}% for Label {}'.format(index / len(input_csv_files) * 100, label))

    feat_arr = np.concatenate(feature, axis=0)
    if save:
        np.savez_compressed("x_test_{}.npz".format(
            label, win_len, int(thrshd * 100), step), feat_arr)
    # one hot
    feat_label = np.zeros((feat_arr.shape[0], len(labels)))
    feat_label[:, labels.index(label)] = 1
    return feat_arr, feat_label


def train_valid_split(numpy_tuple, train_portion=0.8, seed=200):
    """
    Returns Train and Valid Datset with the format of (x_train, y_train, x_valid, y_valid),
    where x_train and y_train are shuffled randomly.

    Args:
        numpy_tuple  : tuple of numpy array: (x_wave, x_clap, x_walk, x_liedown, x_sitdown, x_fall, x_pickup)
        train_portion: float, range (0,1)
        seed         : random seed
    """
    np.random.seed(seed=seed)
    x_train = []
    x_valid = []
    y_valid = []
    y_train = []

    for i, x_arr in enumerate(numpy_tuple):
        index = np.random.permutation([i for i in range(x_arr.shape[0])])
        split_len = int(train_portion * x_arr.shape[0])
        x_train.append(x_arr[index[:split_len], ...])
        tmpy = np.zeros((split_len,7))
        tmpy[:, i] = 1
        y_train.append(tmpy)
        x_valid.append(x_arr[index[split_len:],...])
        tmpy = np.zeros((x_arr.shape[0]-split_len,7))
        tmpy[:, i] = 1
        y_valid.append(tmpy)

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_valid = np.concatenate(x_valid, axis=0)
    y_valid = np.concatenate(y_valid, axis=0)

    index = np.random.permutation([i for i in range(x_train.shape[0])])
    x_train = x_train[index, ...]
    y_train = y_train[index, ...]
    return x_train, y_train, x_valid, y_valid

def extract_csi(raw_folder, labels, save=False, win_len=200, thrshd=0.6, step=50):
    """
    Return List of Array in the format of [X_label1, y_label1, X_label2, y_label2, .... X_Label7, y_label7]
    Args:
        raw_folder: the folder path of raw CSI csv files
        labels    : all the labels existing in the folder
        save      : boolean, choose whether save the numpy array
        win_len   :  integer, window length
        thrshd    :  float,  determine if an activity is strong enough inside a window
        step      :  integer, sliding window by step
    """
    ans = []
    for label in labels:
        feature_arr, label_arr = extract_csi_by_label(raw_folder, label, labels, save, win_len, thrshd, step)
        ans.append(feature_arr)
        ans.append(label_arr)
    return tuple(ans)


class AttenLayer(tf.keras.layers.Layer):
    """
    Attention Layers used to Compute Weighted Features along Time axis
    Args:
        num_state :  number of hidden Attention state

    edited code provided on https://github.com/ludlows
    """

    def __init__(self, num_state, **kw):
        super(AttenLayer, self).__init__(**kw)
        self.num_state = num_state

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel', shape=[input_shape[-1], self.num_state])
        self.bias = self.add_weight('bias', shape=[self.num_state])
        self.prob_kernel = self.add_weight('prob_kernel', shape=[self.num_state])

    def call(self, input_tensor):
        atten_state = tf.tanh(tf.tensordot(input_tensor, self.kernel, axes=1) + self.bias)
        logits = tf.tensordot(atten_state, self.prob_kernel, axes=1)
        prob = tf.nn.softmax(logits)
        weighted_feature = tf.reduce_sum(tf.multiply(input_tensor, tf.expand_dims(prob, -1)), axis=1)
        return weighted_feature

    # for saving the model
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_state': self.num_state, })
        return config


class CSIModelConfig:
    """
    class for Human Activity Recognition ('wave', 'clap', 'walk', 'liedown', 'sitdown', 'fall', 'pickup')
    Using CSI (Channel State Information)

    Args:
        win_len   :  integer (200 default) window length for batching sequence
        step      :  integer (50  default) sliding window by this step
        thrshd    :  float   (0.6  default) used to check if the activity is intensive inside a window
        downsample:  integer >=1 (2 default) downsample along the time axis
    """

    def __init__(self, win_len=200, step=50, thrshd=0.6, downsample=1):
        self._win_len = win_len
        self._step = step
        self._thrshd = thrshd
        self._labels = ("wave", "clap", "walk", "liedown", "sitdown", "fall", "pickup")
        self._downsample = downsample

    def preprocessing(self, raw_folder, save=False):
        """
        Returns the Numpy Array for training within the format of (X_lable1, y_label1, ...., X_label7, y_label7)
        Args:
            raw_folder: the folder containing raw CSI
            save      : choose if save the numpy array
        """
        numpy_tuple = extract_csi(raw_folder, self._labels, save, self._win_len, self._thrshd, self._step)
        if self._downsample > 1:
            return tuple([v[:, ::self._downsample, ...] if i % 2 == 0 else v for i, v in enumerate(numpy_tuple)])
        return numpy_tuple

    def load_csi_data_from_files(self, np_files):
        """
        Returns the Numpy Array for training within the format of (X_lable1, y_label1, ...., X_label7, y_label7)
        Args:
            np_files: ('x_wave.npz', 'x_clap.npz', 'x_walk.npz', 'x_liedown.npz', 'x_sitdown.npz', 'x_fall.npz', 'x_pickup.npz')
        """
        if len(np_files) != 7:
            raise ValueError('There should be 7 numpy files for lie down, fall, bend, run, sitdown, standup, walk.')
        x = [np.load(f)['arr_0'] for f in np_files]
        if self._downsample > 1:
            x = [arr[:, ::self._downsample, :] for arr in x]
        y = [np.zeros((arr.shape[0], len(self._labels))) for arr in x]
        numpy_list = []
        for i in range(len(self._labels)):
            y[i][:, i] = 1
            numpy_list.append(x[i])
            numpy_list.append(y[i])
        return tuple(numpy_list)

    def build_model(self, n_unit_lstm=200, n_unit_atten=400):
        """
        Returns the Tensorflow Model which uses AttenLayer
        """
        if self._downsample > 1:
            length = len(np.ones((self._win_len,))[::self._downsample])
            x_in = tf.keras.Input(shape=(length, 64))
        else:
            x_in = tf.keras.Input(shape=(self._win_len, 64))

        x_tensor = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_unit_lstm, return_sequences=True))(x_in)
        x_tensor = AttenLayer(n_unit_atten)(x_tensor)
        pred = tf.keras.layers.Dense(len(self._labels), activation='softmax')(x_tensor)
        model = tf.keras.Model(inputs=x_in, outputs=pred)
        return model

def train_test_split(numpy_tuple, train_portion=1.0, seed=0):
    """
    Returns Train and Valid Datset with the format of (x_train, y_train, x_valid, y_valid),
    where x_train and y_train are shuffled randomly.

    Args:
        numpy_tuple  : tuple of numpy array: (x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk)
        train_portion: float, range (0,1)

        seed         : random seed
    """
    np.random.seed(seed=seed)
    x_test = []
    y_test = []

    for i, x_arr in enumerate(numpy_tuple):
        index = np.random.permutation([i for i in range(x_arr.shape[0])])
        split_len = int(train_portion * x_arr.shape[0])
        x_test.append(x_arr[index[:split_len], ...])
        tmpy = np.zeros((split_len, 7))
        tmpy[:, i] = 1
        y_test.append(tmpy)

    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    index = np.random.permutation([i for i in range(x_test.shape[0])])
    x_test = x_test[index, ...]
    y_test = y_test[index, ...]
    return x_test, y_test

# load models from filel
def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = 'snapshot_model_energy_' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename,custom_objects={'AttenLayer': AttenLayer})
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = array(yhats)
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    # print(summed.shape)
    # argmax across classes
    result = argmax(summed, axis=1)
    return result

# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
    # select a subset of members
    subset = members[:n_members]
    # make prediction
    yhat = ensemble_predictions(subset, testX)
    cm = confusion_matrix(testy,yhat)
    # calculate accuracy
    return accuracy_score(testy, yhat)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Error! Correct Command: python3 csimodel.py Dataset_folder_path")
    raw_data_folder = sys.argv[0]

    # preprocessing
    cfg = CSIModelConfig(win_len=200, step=50, thrshd=0.6, downsample=1)
    numpy_tuple = cfg.preprocessing('./123-test4/test/', save=True)
    # numpy_tuple = cfg.load_csi_data_from_files(('x_total_wave.npz', 'x_total_clap.npz', 'x_total_walk.npz','x_total_liedown.npz', 'x_total_sitdown.npz', 'x_total_fall.npz', 'x_total_pickup.npz'))
    x_wave, y_wave, x_clap, y_clap, x_walk, y_walk, x_liedown, y_liedown, x_sitdown, y_sitdown, x_fall, y_fall, x_pickup, y_pickup = numpy_tuple

    print(x_wave.shape)
    print(x_clap.shape)
    print(x_walk.shape)
    print(x_liedown.shape)
    print(x_sitdown.shape)
    print(x_fall)
    print(x_pickup)

    x_valid, y_valid = train_test_split(
        (x_wave, x_clap, x_walk, x_liedown, x_sitdown, x_fall, x_pickup),
        train_portion=1.0, seed=0)

    y_valid = np.array([np.argmax(i) for i in y_valid])

    # load models in order
    members = load_all_models(10)
    print('Loaded %d models' % len(members))
    # reverse loaded models so we build the ensemble with the last models first
    members = list(reversed(members))
    # evaluate different numbers of ensembles on hold out set
    single_scores, ensemble_scores = list(), list()
    for i in range(1, len(members) + 1):
        # evaluate model with i members
        ensemble_score = evaluate_n_members(members, i, x_valid, y_valid)
        testy_enc = to_categorical(y_valid)
        _, single_score = members[i - 1].evaluate(x_valid, testy_enc, verbose=0)

        # summarize this step
        print('> %d: single=%.4f, ensemble=%.4f' % (i, single_score, ensemble_score))
        ensemble_scores.append(ensemble_score)
        single_scores.append(single_score)
    # summarize average accuracy of a single final model
    print('Accuracy %.4f (%.4f)' % (mean(single_scores), std(single_scores)))



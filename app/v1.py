import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
import cPickle as pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import provider
import train_util

MODEL_PATH = 'pretrained/log_v1/model.ckpt'
DATA_PATH = 'kitti/frustum_carpedcyc_val_rgb_detection.pickle'

BATCH_SIZE = 1
NUM_POINT = 1024
NUM_CHANNEL = 4

fp_nets = importlib.import_module('frustum_pointnets_v1')


class FPNetPredictor(object):

    graph = tf.Graph()
    sess = None
    saver = None
    ops = None

    def __init__(self, model_fp):
        self.model_fp = model_fp
        with tf.device('/gpu:0'):
            self._init_session()
            self._init_graph()

    def _init_session(self):
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            self.sess = tf.Session(config=config)

    def _init_graph(self):
        with self.graph.as_default():
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = \
                fp_nets.placeholder_inputs(BATCH_SIZE, NUM_POINT)

            is_training_pl = tf.placeholder(tf.bool, shape=())
            end_points = fp_nets.get_model(pointclouds_pl, one_hot_vec_pl, is_training_pl)

            self.saver = tf.train.Saver()
            # Restore variables from disk.
            self.saver.restore(self.sess, self.model_fp)
            self.ops = {'pointclouds_pl': pointclouds_pl,
                   'one_hot_vec_pl': one_hot_vec_pl,
                   'labels_pl': labels_pl,
                   'centers_pl': centers_pl,
                   'heading_class_label_pl': heading_class_label_pl,
                   'heading_residual_label_pl': heading_residual_label_pl,
                   'size_class_label_pl': size_class_label_pl,
                   'size_residual_label_pl': size_residual_label_pl,
                   'is_training_pl': is_training_pl,
                   'logits': end_points['mask_logits'],
                   'center': end_points['center'],
                   'end_points': end_points}

    def predict(self, pc, one_hot_vec):

        _ops = self.ops
        _ep = _ops['end_points']

        feed_dict = {_ops['pointclouds_pl']: pc, _ops['one_hot_vec_pl']: one_hot_vec, _ops['is_training_pl']: False}

        logits, centers, heading_logits, \
        heading_residuals, size_scores, size_residuals = \
        self.sess.run([_ops['logits'], _ops['center'],
                  _ep['heading_scores'], _ep['heading_residuals'],
                  _ep['size_scores'], _ep['size_residuals']],
                 feed_dict=feed_dict)

        # heading_cls = np.argmax(heading_logits, 1) 
        # size_cls = np.argmax(size_scores, 1) 
        # heading_res = np.array([heading_residuals[i, heading_cls[i]] for i in range(pc.shape[0])])
        # size_res = np.vstack([size_residuals[i, size_cls[i], :] for i in range(pc.shape[0])])

        return logits, centers, heading_logits, heading_residuals, size_scores, size_residuals


def test():

    # Load Frustum Datasets.
    TEST_DATASET = provider.FrustumDataset(
        npoints=NUM_POINT,
        split='val',
        rotate_to_center=True,
        overwritten_data_path=DATA_PATH,
        from_rgb_detection=True,
        one_hot=True)

    # Inherit test.py code and use 1 batch
    num_batches = int((len(TEST_DATASET)+BATCH_SIZE-1)/BATCH_SIZE)
    test_idxs = np.arange(0, len(TEST_DATASET))
    batch_data_to_feed = np.zeros((BATCH_SIZE, NUM_POINT, NUM_CHANNEL))
    batch_one_hot_to_feed = np.zeros((BATCH_SIZE, 3))

    for batch_idx in range(num_batches):
        print('batch idx: %d' % (batch_idx))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * BATCH_SIZE)
        cur_batch_size = end_idx - start_idx

        batch_data, batch_rot_angle, batch_rgb_prob, batch_one_hot_vec = \
            train_util.get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL, from_rgb_detection=True)

        # Show how the data look like
        print('batch_data len: {}'.format(len(batch_data)))
        print('batch_data[0]: {}'.format(batch_data[0]))
        print('batch_data[0] type: {}'.format(type(batch_data[0])))
        print('batch_data[0] len: {}'.format(len(batch_data[0])))
        print('batch_data[0][0] len: {}'.format(len(batch_data[0][0])))
        
        print('batch_one_hot_vec len: {}'.format(len(batch_one_hot_vec)))
        print('batch_one_hot_vec[0]: {}'.format(batch_one_hot_vec[0]))

        batch_data_to_feed[0:cur_batch_size,...] = batch_data
        batch_one_hot_to_feed[0:cur_batch_size,:] = batch_one_hot_vec

    # Data to feed: 1024 points [[x y z int]...[]] and one hot vector [0. 0. 1.] 
    pc = batch_data_to_feed
    one_hot_vec = batch_one_hot_to_feed
    print 'len of point cloud', len(pc)
    print 'len of one_hot_vec', len(one_hot_vec)

    # Demo how to use this predictor
    predictor = FPNetPredictor(model_fp=MODEL_PATH)
    print predictor.predict(pc=pc, one_hot_vec=one_hot_vec)

if __name__ == "__main__":
    test()

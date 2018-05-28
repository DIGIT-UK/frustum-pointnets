import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
import cPickle as pickle

from train import provider, train_util

MODEL_PATH = './'
DATA_PATH = ''

'''
Unfortunately, in models/model_util, the BATCH_SIZE is hardcoded into tf placeholder,
therefore, you have to retrain your own model with placeholder first dimension set to be None.
Changes required.
I iterated every nodes in pre-trained model to have verified this facts.
So, that's said, with provided pre-trained model, BATCH_SIZE has to be 32
Same as NUM_POINT to be 1024
'''
BATCH_SIZE = 32
NUM_POINT = 1024
NUM_CHANNEL = 4
NUM_CLASSES = 2
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8 # one cluster for each type
NUM_OBJECT_POINT = 512
fp_nets = importlib.import_module('models.frustum_pointnets_v1')

with tf.Graph().as_default():
    with tf.device('/gpu:0'):

        pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
        heading_class_label_pl, heading_residual_label_pl, \
        size_class_label_pl, size_residual_label_pl = \
            fp_nets.placeholder_inputs(BATCH_SIZE, NUM_POINT)

        is_training_pl = tf.placeholder(tf.bool, shape=())

        end_points = fp_nets.get_model(pointclouds_pl, one_hot_vec_pl, is_training_pl)

        loss = fp_nets.get_loss(labels_pl, centers_pl,
                              heading_class_label_pl, heading_residual_label_pl,
                              size_class_label_pl, size_residual_label_pl, end_points)
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    ops = {'pointclouds_pl': pointclouds_pl,
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
           'end_points': end_points,
           'loss': loss}


def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs


def inference(sess, ops, pc, one_hot_vec, batch_size):
    ''' Run inference for frustum pointnets in batch mode '''
    assert pc.shape[0] % batch_size == 0
    num_batches = pc.shape[0] / batch_size
    logits = np.zeros((pc.shape[0], pc.shape[1], NUM_CLASSES))
    centers = np.zeros((pc.shape[0], 3))
    heading_logits = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    heading_residuals = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    size_logits = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER))
    size_residuals = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER, 3))
    scores = np.zeros((pc.shape[0],))  # 3D box score

    ep = ops['end_points']
    for i in range(num_batches):
        feed_dict = { \
            ops['pointclouds_pl']: pc[i * batch_size:(i + 1) * batch_size, ...],
            ops['one_hot_vec_pl']: one_hot_vec[i * batch_size:(i + 1) * batch_size, :],
            ops['is_training_pl']: False}

        batch_logits, batch_centers, \
        batch_heading_scores, batch_heading_residuals, \
        batch_size_scores, batch_size_residuals = \
            sess.run([ops['logits'], ops['center'],
                      ep['heading_scores'], ep['heading_residuals'],
                      ep['size_scores'], ep['size_residuals']],
                     feed_dict=feed_dict)

        logits[i * batch_size:(i + 1) * batch_size, ...] = batch_logits
        centers[i * batch_size:(i + 1) * batch_size, ...] = batch_centers
        heading_logits[i * batch_size:(i + 1) * batch_size, ...] = batch_heading_scores
        heading_residuals[i * batch_size:(i + 1) * batch_size, ...] = batch_heading_residuals
        size_logits[i * batch_size:(i + 1) * batch_size, ...] = batch_size_scores
        size_residuals[i * batch_size:(i + 1) * batch_size, ...] = batch_size_residuals

        # Compute scores
        batch_seg_prob = softmax(batch_logits)[:, :, 1]  # BxN
        batch_seg_mask = np.argmax(batch_logits, 2)  # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1)  # B,
        mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask, 1)  # B,
        heading_prob = np.max(softmax(batch_heading_scores), 1)  # B
        size_prob = np.max(softmax(batch_size_scores), 1)  # B,
        batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
        scores[i * batch_size:(i + 1) * batch_size] = batch_scores
        # Finished computing scores

    heading_cls = np.argmax(heading_logits, 1)  # B
    size_cls = np.argmax(size_logits, 1)  # B
    heading_res = np.array([heading_residuals[i, heading_cls[i]] \
                            for i in range(pc.shape[0])])
    size_res = np.vstack([size_residuals[i, size_cls[i], :] \
                          for i in range(pc.shape[0])])

    return np.argmax(logits, 2), centers, heading_cls, heading_res, \
           size_cls, size_res, scores


# Load Frustum Datasets.
TEST_DATASET = provider.FrustumDataset(
    npoints=NUM_POINT,
    split='val',
    rotate_to_center=True,
    overwritten_data_path=DATA_PATH,
    one_hot=True)

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
    batch_data_to_feed[0:cur_batch_size,...] = batch_data
    batch_one_hot_to_feed[0:cur_batch_size,:] = batch_one_hot_vec

    # Run one batch inference
batch_output, batch_center_pred, \
batch_hclass_pred, batch_hres_pred, \
batch_sclass_pred, batch_sres_pred, batch_scores = \
    inference(sess, ops, batch_data_to_feed,
              batch_one_hot_to_feed, batch_size=BATCH_SIZE)
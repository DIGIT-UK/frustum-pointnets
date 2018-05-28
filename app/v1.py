import os
import sys
import importlib
import numpy as np
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import provider

MODEL_PATH = 'pretrained/log_v1/model.ckpt'
DATA_PATH = 'kitti/frustum_carpedcyc_val_rgb_detection.pickle'

BATCH_SIZE = 1
NUM_POINT = 1024
NUM_CHANNEL = 4

fp_nets = importlib.import_module('frustum_pointnets_v1')
tf.logging.set_verbosity(tf.logging.INFO)

class FPNetPredictor(object):

    graph = tf.Graph()
    sess = None
    saver = None
    ops = None

    def __init__(self, model_fp):
        tf.logging.info("Initializing FPNetPredictor Instance ...")
        self.model_fp = model_fp
        with tf.device('/gpu:0'):
            self._init_session()
            self._init_graph()
        tf.logging.info("Initialized FPNetPredictor Instance!")

    def _init_session(self):
        tf.logging.info("Initializing Session ...")
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            self.sess = tf.Session(config=config)

    def _init_graph(self):
        tf.logging.info("Initializing Graph ...")
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
        tf.logging.info("Predicting with pointcloud and one hot vector ...")
        _ops = self.ops
        _ep = _ops['end_points']

        feed_dict = {_ops['pointclouds_pl']: pc, _ops['one_hot_vec_pl']: one_hot_vec, _ops['is_training_pl']: False}

        logits, centers, heading_logits, \
        heading_residuals, size_scores, size_residuals = \
        self.sess.run([_ops['logits'], _ops['center'],
                  _ep['heading_scores'], _ep['heading_residuals'],
                  _ep['size_scores'], _ep['size_residuals']],
                 feed_dict=feed_dict)

        tf.logging.info("Prediction done ! \nResults:\nCenter: {}\nSize Score: {}".format(centers, size_scores))
        return logits, centers, heading_logits, heading_residuals, size_scores, size_residuals


def test():

    # Load Frustum Datasets.
    print 'Loading data .....'
    TEST_DATASET = provider.FrustumDataset(
        npoints=NUM_POINT,
        split='val',
        rotate_to_center=True,
        overwritten_data_path=DATA_PATH,
        from_rgb_detection=True,
        one_hot=True)
    print 'Data loaded !'
    # Select one of the datasets as test input to the NN
    one_test_data = TEST_DATASET[0]
    pc = one_test_data[0]
    rot_angle = one_test_data[1]
    prob_list = one_test_data[2]
    one_hot_vec = one_test_data[-1]
    print 'Test data: '
    print '     Point Cloud Input: ', pc
    print '     One Hot Vector Input: ', one_hot_vec
    
    print 'Auxiliary data: '
    print '     Rot Angle: ', rot_angle
    print '     Prob List: ', prob_list

    # Data to feed: 1024 points [[x y z int]...[]] and one hot vector [0. 0. 1.] 
    print '     len of point cloud', len(pc)
    print '     len of one_hot_vec', len(one_hot_vec)

    # Demo how to use this predictor
    predictor = FPNetPredictor(model_fp=MODEL_PATH)
    predictor.predict(pc=[pc], one_hot_vec=[one_hot_vec])

    
if __name__ == "__main__":
    test()

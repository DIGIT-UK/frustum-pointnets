import numpy as np
import v1
import frustum_proposal 
import provider

# TODO: Clean up!!!

def viz(pc, centers, corners_3d, pc_origin):
    import mayavi.mlab as mlab
    fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
        fgcolor=None, engine=None, size=(500, 500))
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], mode='sphere',
        colormap='gnuplot', scale_factor=0.1, figure=fig)
    mlab.points3d(centers[:,0], centers[:,1], centers[:,2], mode='sphere',
        color=(1, 0, 1), scale_factor=0.3, figure=fig)
    mlab.points3d(corners_3d[:,0], corners_3d[:,1], corners_3d[:,2], mode='sphere',
        color=(1, 1, 0), scale_factor=0.3, figure=fig)
    mlab.points3d(pc_origin[:,0], pc_origin[:,1], pc_origin[:,2], mode='sphere',
        color=(0, 1, 0), scale_factor=0.05, figure=fig)
    '''
        Green points are original PC from KITTI
        White points are PC feed into the network
        Red point is the predicted center
        Yellow point the post-processed predicted bounding box corners
    '''
    raw_input("Press any key to continue")

def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

# Ugly demo with raw data from trainig/000000

BATCH_SIZE = 1
NUM_POINT = 1024
NUM_CHANNEL = 4
NUM_HEADING_BIN = 12
MODEL_PATH = 'pretrained/log_v1/model.ckpt'
pc_vel = load_velo_scan('dataset/KITTI/object/training/velodyne/000000.bin')

# You can find below hardcoded number in /KITTI/object/training/*/000000.*
clbs = {
	'id': '000000',
	'P2': (7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02, 4.575831000000e+01, 0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02, -3.454157000000e-01, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03),
	'Tr_velo_to_cam': (6.927964000000e-03, -9.999722000000e-01, -2.757829000000e-03, -2.457729000000e-02, -1.162982000000e-03, 2.749836000000e-03, -9.999955000000e-01, -6.127237000000e-02, 9.999753000000e-01, 6.931141000000e-03, -1.143899000000e-03, -3.321029000000e-01),
	'R0_rect': (9.999128000000e-01, 1.009263000000e-02, -8.511932000000e-03, -1.012729000000e-02, 9.999406000000e-01, -4.037671000000e-03, 8.470675000000e-03, 4.123522000000e-03, 9.999556000000e-01)
}

img_shape = (1224, 370)
bbs = [(712.40, 143.00, 810.73, 307.92)]

# Step1, get frustum proposals with image_shape, predicted bounding box and original point cloud
test = frustum_proposal.FrustumProposal(clbs)
pc = test.get_frustum_proposal(img_shape, bbs, pc_vel)[0]
one_hot_vec = (1, 0, 0)


# Downsample points into 1024
choice = np.random.choice(pc.shape[0], NUM_POINT, replace=True)
pc = pc[choice, :]

# Step2, predict 3d bounding box with frustum proposals
predictor = v1.FPNetPredictor(model_fp=MODEL_PATH)

logits, centers, \
heading_logits, heading_residuals, \
size_scores, size_residuals = predictor.predict(pc=[pc], one_hot_vec=[one_hot_vec])

# Step3, Post process

# Get 3D bounding box
heading_class = np.argmax(heading_logits, 1)
size_logits = size_scores
size_class = np.argmax(size_logits, 1) 
size_residual = np.vstack([size_residuals[0,size_class[0],:]])
heading_residual = np.array([heading_residuals[0,heading_class[0]]]) # B,
heading_angle = provider.class2angle(heading_class[0],heading_residual[0], NUM_HEADING_BIN)
box_size = provider.class2size(size_class[0], size_residual[0])
corners_3d = provider.get_3d_box(box_size, heading_angle, centers[0])


# Step4, Visualization
viz(pc, centers, corners_3d, pc_vel)

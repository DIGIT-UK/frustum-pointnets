import numpy as np
import frustum_point_net
import frustum_proposal 
import provider
import tools


# Ugly demo with raw data from trainig/000000

BATCH_SIZE = 1
NUM_POINT = 1024
NUM_CHANNEL = 4
NUM_HEADING_BIN = 12
MODEL_PATH = 'pretrained/log_v1/model.ckpt'
pc_vel = tools.load_velo_scan('dataset/KITTI/object/training/velodyne/000000.bin')
calib_f = tools.read_calib_file('dataset/KITTI/object/training/calib/000000.txt')
img_shape = (1224, 370)
# You can find below hardcoded number in /KITTI/object/training/*/000000.*
bbs = [(712.40, 143.00, 810.73, 307.92)]
clbs = {
	'id': '000000',
	'P2': calib_f['P2'],
	'Tr_velo_to_cam': calib_f['Tr_velo_to_cam'],
	'R0_rect': calib_f['R0_rect']
}


# Step1, get frustum proposals with image_shape, predicted bounding box and original point cloud
fpnet = frustum_proposal.FrustumProposal(clbs)
fp_pc, fp_pc_velo = fpnet.get_frustum_proposal(img_shape, bbs, pc_vel)
fp_pc = fp_pc[0]
fp_pc_velo = fp_pc_velo[0]
one_hot_vec = (0, 1, 0)


# Downsample points into 1024
choice = np.random.choice(fp_pc.shape[0], NUM_POINT, replace=True)
fp_pc = fp_pc[choice, :]

# Step2, predict 3d bounding box with frustum proposals
predictor = frustum_point_net.FPNetPredictor(model_fp=MODEL_PATH)

logits, centers, \
heading_logits, heading_residuals, \
size_scores, size_residuals = predictor.predict(pc=[fp_pc], one_hot_vec=[one_hot_vec])

# Step3, Post process ---> Get 3D bounding box from tensorflow raw output
heading_class = np.argmax(heading_logits, 1)
size_logits = size_scores
size_class = np.argmax(size_logits, 1) 
size_residual = np.vstack([size_residuals[0,size_class[0],:]])
heading_residual = np.array([heading_residuals[0,heading_class[0]]]) # B,
heading_angle = provider.class2angle(heading_class[0],heading_residual[0], NUM_HEADING_BIN)
box_size = provider.class2size(size_class[0], size_residual[0])
corners_3d = provider.get_3d_box(box_size, heading_angle, centers[0])

corners_3d_in_velo_frame = np.zeros_like(corners_3d)
centers_in_velo_frame = np.zeros_like(centers)
corners_3d_in_velo_frame[:, 0:3] = fpnet.project_rect_to_velo(corners_3d[:, 0:3])
centers_in_velo_frame[:, 0:3] = fpnet.project_rect_to_velo(centers[:, 0:3])

# Step4, Visualization
tools.viz(fp_pc_velo, centers_in_velo_frame, corners_3d_in_velo_frame, pc_vel)

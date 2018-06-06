import numpy as np
import mayavi.mlab as mlab


def viz(pc, centers, corners_3d, pc_origin):
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


def read_calib_file(filepath):
    ''' Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    '''
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line)==0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data
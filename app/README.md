# Intro

Due to this project used [tf.py_func](https://www.tensorflow.org/api_docs/python/tf/py_func)
twice, plus using BatchNorm, it is not easy to freeze the model and deploy it without
too much efforts on other platforms.

Therefore, we directly utilize the pre-trained model to build an app that can process pointcloud/camera/camera_info
predicting 3D objects.


# Summary on FPNet

FPNet has three steps:

- [X] 2D Image object detection (input image ---> output bounding box)

- [X] [Frustum Proposals](https://github.com/Dark-Rinnegan/frustum-pointnets/blob/app/app/frustum_proposal.py) (input predictions from step1 and camera-lidar calibrations info ----> output frustum point cloud proposals)

- [X] [3D Bounding Box predictions](https://github.com/Dark-Rinnegan/frustum-pointnets/blob/app/app/frustum_point_net.py) (input frustum proposal ----> output 3D bounding boxes)

Under app folder, we only contains demo for step2&3 since step 1 is too easy (if you are a newbie, 
try [this](https://github.com/KleinYuan/tf-object-detection) or [this](https://github.com/Dark-Rinnegan/keras-yolo3)).



# How to run it

### For step 2 only:

- [X] Download pretrained model and put it under pretrained folder

- [X] Prepare KITTI data and ensure `frustum_carpedcyc_val_rgb_detection.pickle` is under kitti folder

- [X] Run `python app/frustum_point_net.py`

Then you should be able to see:

![predict](https://user-images.githubusercontent.com/8921629/40632268-d2f4bc5e-6299-11e8-8248-48d42748e6b9.png)


### For step 2 and 3 together:

- [X] Config the file path [here](https://github.com/Dark-Rinnegan/frustum-pointnets/blob/app/app/demo.py#L15)

- [X] Fetch the bounding box param from KITTI labels txt file or if you hook up your own 2d image detector, use that info, 
in [here](https://github.com/Dark-Rinnegan/frustum-pointnets/blob/app/app/demo.py#L19). Same order

- [X] Encode the class in label in this [line](https://github.com/Dark-Rinnegan/frustum-pointnets/blob/app/app/demo.py#L35)

- [X] Run `python app/demo.py`

Then you should be able to see:

![semi-endtoend](https://user-images.githubusercontent.com/8921629/41068890-76807090-69a0-11e8-9794-62fc394667b3.png)

corresponding to this image:

![000003](https://user-images.githubusercontent.com/8921629/41068960-bdc9d78e-69a0-11e8-82f7-48c92786811e.png)


# TODO

- [X] Adding frustum proposal part

- [ ] Hookup with 2D object detectors such as [this](https://github.com/KleinYuan/tf-object-detection) or [this](https://github.com/Dark-Rinnegan/keras-yolo3))

- [ ] Rewrite py_func part so that this graph can be frozen
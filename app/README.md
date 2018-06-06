# Intro

Due to this project used [tf.py_func](https://www.tensorflow.org/api_docs/python/tf/py_func)
twice, plus using BatchNorm, it is not easy to freeze the model and deploy it without
too much efforts on other platforms.

Therefore, we directly utilize the pre-trained model to build an app that can process pointcloud/camera/camera_info
predicting 3D objects.

# How to run it

- [X] Download pretrained model and put it under pretrained folder

- [X] Prepare KITTI data and ensure `frustum_carpedcyc_val_rgb_detection.pickle` is under kitti folder

- [X] Run `python app/frustum_point_net.py`

Then you should be able to see:

![predict](https://user-images.githubusercontent.com/8921629/40632268-d2f4bc5e-6299-11e8-8248-48d42748e6b9.png)


# TODO

- [X] Adding frustum proposal part

- [ ] Rewrite py_func part so that this graph can be frozen
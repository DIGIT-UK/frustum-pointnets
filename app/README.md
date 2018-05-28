# Intro

Due to this project used [tf.py_func](https://www.tensorflow.org/api_docs/python/tf/py_func)
twice, plus using BatchNorm, it is not easy to freeze the model and deploy it without
too much efforts on other platforms.

Therefore, we directly utilize the pre-trained model to build an app that can process pointcloud/camera/camera_info
predicting 3D objects.

import numpy as np


class Camera(object):

    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat

        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


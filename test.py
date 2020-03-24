import torch

import numpy as np

from lib.model.rpn.generate_anchors import generate_anchors

_anchors = torch.from_numpy(generate_anchors(scales=np.array([8, 16, 32]),
            ratios=np.array([0.5, 1, 2]))).float()

_anchors = _anchors.view(1, 9, 4)

feat_width = 10
feat_height = 8
_feat_stride = 100

shift_x = np.arange(0, feat_width) * _feat_stride
shift_y = np.arange(0, feat_height) * _feat_stride
shift_x, shift_y = np.meshgrid(shift_x, shift_y)
shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
shifts = shifts.view(feat_width*feat_height, 1, 4).float()
anchors = _anchors + shifts
anchors = anchors.view(1, 9*feat_width*feat_height, 4)

import cv2
image = cv2.imread('lebron.jpg')
for i in range(0, feat_height * feat_width * 9):
    xmin = int(anchors[0][i][0])
    ymin = int(anchors[0][i][1])
    xmax = int(anchors[0][i][2])
    ymax = int(anchors[0][i][3])

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
cv2.imwrite('2.jpg', image)
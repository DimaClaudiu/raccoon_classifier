#! /usr/bin/env python

import sys
from classifier import *


def main(path):
    # Set parameters
    net_h, net_w = 416, 416  # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    # Load the model
    infer_model = load_model('raccoon.h5', compile=False)

    # Do detection on an image
    image = cv2.imread(path)

    # Predict the bounding boxes
    score = get_yolo_score(infer_model, [
        image], net_h, net_w, [17, 18, 28, 24, 36, 34, 42, 44, 56, 51, 72, 66, 90, 95, 92, 154, 139, 281], obj_thresh, nms_thresh)

    # Print the scores
    print(score)


if __name__ == '__main__':
    main(sys.argv[1])

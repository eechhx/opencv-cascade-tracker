#!/usr/bin/env python3

import cv2 as cv
import sys

class tracker_types:
    CSRT = cv.TrackerCSRT_create()
    KCF = cv.TrackerKCF_create()
    MEDIANFLOW = cv.TrackerMedianFlow_create()
    
    def __init__(self):
        pass

if __name__ == '__main__':
    pass
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import argparse as ap
import numpy as np
import cv2 as cv
import os
import sys

class CustomFormatter(ap.HelpFormatter):
    def _format_action_invocation(self, action):
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            if action.nargs == 0:
                parts.extend(action.option_strings)
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    #parts.append('%s %s' % (option_string, args_string))
                    parts.append('%s' % option_string)
                parts[-1] += ' %s'%args_string
            return ', '.join(parts)

# Parser Arguments
parser = ap.ArgumentParser(description='Cascade Classifier', formatter_class=CustomFormatter)
parser.add_argument("-s", "--save", metavar='', help="specify output name")
parser.add_argument("-c", "--cas", metavar='', help="specify specific trained cascade", default="./stage_outputs/cascade.xml")
parser.add_argument("-i", "--img", metavar='', help="specify image to be classified")
parser.add_argument("-d", "--dir", metavar='', help="specify directory of images to be classified")
parser.add_argument("-v", "--vid", metavar='', help="specify video to be classified")
parser.add_argument("-w", "--cam", metavar='', help="enable camera access for classification")
parser.add_argument("-f", "--fps", help="enable frames text (TODO)", action="store_true")
parser.add_argument("-o", "--circle", help="enable circle detection", action="store_true")
parser.add_argument("-z", "--scale", metavar='', help="decrease video scale by scale factor", type=int, default=1)
parser.add_argument("-t", "--track", metavar='', help="select tracking algorithm [KCF, CSRT, MEDIANFLOW]", choices=['KCF', 'CSRT', 'MEDIANFLOW'])
args = parser.parse_args(sys.argv[1:])

# Load the trained cascade
cascade = cv.CascadeClassifier()
if not cascade.load(args.cas):
    print("Can't find cascade file. Do you have the directory ./stage_outputs/cascade.xml")
    exit(0)

def plot():
    pass

def detect_circles(src):
    img = src
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.medianBlur(img_gray, 5)
    rows = img_blur.shape[0]
    #Images circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, rows / 3, param1=100, param2=40, maxRadius=40)
    circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, rows/3, param1=100, param2=15, minRadius=10, maxRadius=15)  

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(img, center, radius, (255, 0, 255), 3)
    
    return img

def choose_tracker():
    OPENCV_TRACKERS = {
        'KCF': cv.TrackerKCF_create(),
        'CSRT': cv.TrackerCSRT_create(),
        'MEDIANFLOW': cv.TrackerMedianFlow_create()
    }
    tracker = OPENCV_TRACKERS[args.track]
    return tracker

def tracking(vid, tracker):
    ok, frame = vid.read()
    frame = scale(frame, args.scale)
    ok, roi = tracker.update(frame)

    if ok:
        p1 = (int(roi[0]), int(roi[1]))
        p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
        cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    return frame

def save(frame):
    # Need dimensions of frame to determine proper video output
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    height, width, channels = frame.shape
    out = cv.VideoWriter(args.save + '.avi', fourcc, 30.0, (width, height))
    return out

def get_roi(frame):
    # Get initial bounding box by running cascade detection on first frame
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.GaussianBlur(frame_gray, (3, 3), 0)
    cas_object = cascade.detectMultiScale(frame_gray, minNeighbors=10)
    roi = (cas_object[0][0], cas_object[0][1], cas_object[0][2], cas_object[0][3])
    return roi

def get_cascade(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.GaussianBlur(frame_gray, (3, 3), 0)
    cas_object = cascade.detectMultiScale(frame_gray, minNeighbors=10)
    for (x, y, w, h) in cas_object:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
    return frame

def scale(frame, scale_factor):
    height, width, channels = frame.shape
    scaled_height = int(height/scale_factor)
    scaled_width = int(width/scale_factor)
    resized_frame = cv.resize(frame, (scaled_width, scaled_height))
    return resized_frame

def img_classifier():
    # Read image, convert to gray, equalize histogram, and detect.
    img = cv.imread(args.img, cv.IMREAD_COLOR)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #img_gray = cv.equalizeHist(img_gray)
    cas_object = cascade.detectMultiScale(img_gray, minNeighbors=3, minSize=(200, 200))

    for (x, y, w, h) in cas_object:
        roi = cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

        if args.circle is True:
            roi = img[y:y+h, x:x+w]
            img = detect_circles(roi)

    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def dir_classifier():
    imgs = []
    for filename in os.listdir(args.dir):
        img = cv.imread(os.path.join(args.dir, filename))
        if img is not None:
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            #img_gray = cv.equalizeHist(img_gray)
            cas_object = cascade.detectMultiScale(img_gray, minNeighbors=10)

            for (x, y, w, h) in cas_object:
                cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

                if args.circle is True:
                    roi = img[y:y+h, x:x+w]
                    img = detect_circles(roi)

            cv.imshow(str(filename), img)
            cv.waitKey(0)
            imgs.append(img)
        #print(imgs)
        #return imgs

def vid_classifier():
    vid = cv.VideoCapture(args.vid)

    if not vid.isOpened():
        print("Could not open video")
        sys.exit()
    
    # Read the first frame 
    _ , frame = vid.read()
    frame = scale(frame, args.scale)

    if not _:
        print("Cannot read video file")
        sys.exit()

    if args.save is not None and _ is True:
        out = save(frame=frame)

    if args.track is not None and _ is True:
        process_frame = get_roi(frame)
        tracker = choose_tracker()
        tracker.init(frame, process_frame)
 
    while(vid.isOpened()):
        _ , frame = vid.read()
        frame = scale(frame, args.scale)
        frame = get_cascade(frame)
        
        if args.track is not None:
            frame = tracking(vid=vid, tracker=tracker)

        if args.circle is True:
            roi_circle = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
            frame = detect_circles(roi_circle)    
        
        cv.imshow('video', frame)
        if args.save is not None:
            out.write(frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    if args.save is not None:
        out.release()
    vid.release()
    cv.destroyAllWindows()

def cam_classifier():
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        raise IOError("Cannot access camera")

    while(cam.isOpened()):
        _, frame = cap.read()
        cam_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cam_gray = cv.GaussianBlur(cam_gray, (3, 3), 0)
        cas_object = cascade.detectMultiScale(cam_gray)

        for (x, y, w, h) in cas_object:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)

        cv2.imshow('camera', frame)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    if args.img is not None:
        img_classifier()
    elif args.vid is not None:
        vid_classifier()
    elif args.dir is not None:
        dir_classifier()
    elif args.cam is not None:
        cam_clasifier()
    else:
        parser.print_help()
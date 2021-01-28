#!/usr/bin/env python

import matplotlib.pyplot as plt
import argparse as ap
import numpy as np
import cv2 as cv
import os

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
parser.add_argument("-o", "--cir", help="enable circle detection", action="store_true")
args = parser.parse_args()

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
    #Images circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, rows / 3, param1=100, param2=40)
    circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, rows / 4, param1=100, param2=40, maxRadius=40)  

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

def img_classifier():
    # Read image, convert to gray, equalize histogram, and detect.
    img = cv.imread(args.img, cv.IMREAD_COLOR)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #img_gray = cv.equalizeHist(img_gray)
    cas_object = cascade.detectMultiScale(img_gray, minNeighbors=3, minSize=(200, 200))

    for (x, y, w, h) in cas_object:
        roi = cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

        if args.cir is True:
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

                if args.cir is True:
                    roi = img[y:y+h, x:x+w]
                    img = detect_circles(roi)

            cv.imshow(str(filename), img)
            cv.waitKey(0)
            imgs.append(img)
        #print(imgs)
        #return imgs

def vid_classifier():
    vid = cv.VideoCapture(args.vid)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    _ , frame = vid.read()

    if args.save is not None and _ is True:
        # Need dimensions of frame to get proper video output
        height, width, channels = frame.shape
        out = cv.VideoWriter(args.save + '.avi', fourcc, 20.0, (width, height))

    while(vid.isOpened()):
        # read() returns a tuple; _ indicates if frame was obtained with success 
        _ , frame = vid.read()

        width = vid.get(3)   # float `width`
        height = vid.get(4)  # float `height`

        # Convert BGR to HSV
        if _ is True:
            vid_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vid_gray = cv.GaussianBlur(vid_gray, (3, 3), 0)
            cas_object = cascade.detectMultiScale(vid_gray, minNeighbors=10, minSize=(400, 400))

            for (x, y, w, h) in cas_object:
                roi = cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)

                if args.cir is True:
                    roi = frame[y:y+h, x:x+w]
                    frame = detect_circles(roi)

            cv.imshow('video', frame)

            if args.save is not None:
                out.write(frame)

            if cv.waitKey(40) & 0xFF == ord('q'):
                break
        else:
            break

    vid.release()

    if args.save is not None:
        out.release()
    
    cv.destroyAllWindows()

def cam_classifier():
    cam = cv.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cam.isOpened():
        raise IOError("Cannot access camera")

    while(cam.isOpened()):
        _, frame = cap.read()

        cam_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cam_gray = cv.GaussianBlur(vid_gray, (3, 3), 0)
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
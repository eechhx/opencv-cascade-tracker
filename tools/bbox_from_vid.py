#!/usr/bin/env python3

import argparse as ap
import numpy as np
import cv2 as cv
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
parser = ap.ArgumentParser(description='Get bbox / ROI coords and training images from videos', formatter_class=CustomFormatter)
parser.add_argument("-v", "--vid", metavar='', help="specify video to be loaded")
parser.add_argument("-o", "--center", help="select bounding box / ROI from center point", action='store_true')
parser.add_argument("-c", "--csv", metavar='', help="export CSV file with bbox coords")
parser.add_argument("-z", "--scale", metavar='', help="decrease video scale by scale factor", type=int, default=1)
args = parser.parse_args(sys.argv[1:])

class tracker_types:
    CSRT = cv.TrackerCSRT_create()
    KCF = cv.TrackerKCF_create()
    MEDIANFLOW = cv.TrackerMedianFlow_create()
    
    def __init__(self):
        pass

def scale(frame, scale_factor):
    height, width, channels = frame.shape
    scaled_height = int(height/scale_factor)
    scaled_width = int(width/scale_factor)
    resized_frame = cv.resize(frame, (scaled_width, scaled_height))
    return resized_frame

def create_csv(values):
    np.savetxt(args.csv, values, delimiter=',', fmt='%s')

if __name__ == '__main__':
    if args.vid is not None:
        vid = cv.VideoCapture(args.vid)

        if not vid.isOpened():
            print("Could not open video")
            sys.exit()

        _, frame = vid.read()
        frame = scale(frame, args.scale)
        if not _:
            print("Cannot read video file")
            sys.exit()

        bbox = cv.selectROI(frame, showCrosshair=True, fromCenter=args.center)
        csv_values = np.array([["x_min", "y_min", "x_max", "y_max", "frame_num"]])
        tracker = tracker_types.CSRT
        tracker.init(frame, bbox)

        while True:
            _, frame = vid.read()
            frame = scale(frame, args.scale)
            frame_number = vid.get(cv.CAP_PROP_POS_FRAMES)
            _, roi = tracker.update(frame)

            if _:
                p1 = (int(roi[0]), int(roi[1]))
                p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
                cv.rectangle(frame, p1, p2, (0,255,0), 2, 1)
                cpoint_circle = cv.circle(frame, (int(roi[0]+(roi[2]/2)), int(roi[1]+(roi[3]/2))), 3, (0,255,0), 3)
                csv_data = np.array([[int(roi[0]), int(roi[1]), int(roi[0] + roi[2]), int(roi[1] + roi[3]), int(frame_number)]])
                # If your object is stationary, and you just want to train different lighting conditions
                # csv_data = np.array([[441, 328, 612, 482, int(frame_number)]])
                csv_values = np.append(csv_values, csv_data, 0)
                create_csv(csv_values)
            else:
                # Tracking failure
                cv.putText(frame, "Tracking Failure", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

            # Display result
            cv.imshow("Video", frame)

            # Quit with "Q"
            if cv.waitKey(1) & 0xFF == ord('q'):
                        break

    else:
        parser.print_help()
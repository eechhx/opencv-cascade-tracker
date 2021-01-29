# Training, Classifying (Haar Cascades), and Tracking
Training our own Haar Cascade in OpenCV with Python. A cascade classifier has multiple stages of filtration. Using a sliding window approach, the image region within the window goes through the cascade. Can easily test accuracy of cascade with `classifier.py` script, which takes single images, directory of images, videos, and camera inputs. However, we want to also track our ROI (region of interest). This is because detection (through cascades, etc) is in general, more consuming and computationally taxing. Tracking algorithms are generally considered less taxing, because you know a lot about the apperance of the object in the ROI already. Thus, in the next frame, you use the information from the previous frame to predict and localize where the object is in the frames after.

There are many different types of tracking algorithms that are available through `opencv_contrib/tracking`, such as KCF, MOSSE, TLD, CSRT, etc. Here's a [good video](https://www.youtube.com/watch?v=61QjSz-oLr8) that demonstrates some of these tracking algorithms. Depending on your use case, the one you choose will differ.

## Jump to Section

* [Prerequisites](#prerequisites)
* [Image Scraping](#image_scraping)
* [Postive & Negative Image Sets](#positive-&-negative-image-sets)
* [Positive Samples Image Augmentation](#positive-samples-image-augmentation)
* [Training](#training)
* [Testing Cascade](#testing-cascade)
* [Video Conversions](#video-conversions)
* [Acknowledgements](#acknowledgements)
* [References](#references)

## Prerequisites
* Ubuntu 18.04 
* OpenCV 4.4.0
* OpenCV Contrib (branch parallel with OpenCV 4.4.0)

I used a Python virtual environment for package management. You can build and install OpenCV from source in the virtual environment (especially if you want a specific development branch or full control of compile options), or you can use `pip` locally in the `venv`. Packages included are shown in the `requirements.txt` file for reproducing the specific environment.


## Image Scraping
Go ahead and web scrape relevant negative images for training. Once you have a good amount, filter extensions that aren't `*.jpg` or `*.png` such as `*.gif`. Afterwards, we'll convert all the `*.png` images to `*jpg` using the following command:

```
mogrify -format jpg *.png
```

Then we can delete these `*.png` images. Let's also rename all the images within the directoy to be `img.jpg`

```
ls | cat -n | while read n f; do mv "$f" "img$n.jpg"; done
```

To check if all files within our directory are valid `*.jpg` files:

```
find -name '*.jpg' -exec identify -format "%f" {} \; 1>pass.txt 2>errors.txt
```


## Positive & Negative Image Sets
Positive images correspond to images with detected objects. Images were cropped to 150 x 150 px training set. Negative images are images that are visually close to positive images, but *must not have* any positive image sets within.

<blockquote>

```
/images
    img1.png
    img2.png
positives.txt
```

</blockquote>

To generate your `*.txt` file, run the following command, make sure to change image extension to whatever file type you're using.
```
find ./positive_images -iname "*.png" > positives.txt
```

As a quote from OpenCV docs:
<blockquote>

Negative samples are taken from arbitrary images. These images must not contain detected objects. [...] Described images may be of different sizes. But each image should be (but not nessesarily) larger then a training window size, because these images are used to subsample negative image to the training size.

</blockquote>

## Positive Samples Image Augmentation
We need to create a whole bunch of image samples, and we'll be using OpenCV to augment these images. To create a training set as a collection of PNG images:

```
opencv_createsamples -img ~/ocv-haar-cscade/positive_images/img1.png -bg ~/ocv-haar-cscade/negatives.txt -info ~/ocv-haar-cscade/annotations/annotations.lst -pngoutput -maxxangle 0.1 -maxyangle 0.1 -maxzangle 0.1
```

To augment a set of positive samples with negative samples, let's run the perl script that Naotoshi Seo wrote:
```
perl bin/createsamples.pl positives.txt negatives.txt samples 1500\
  "opencv_createsamples -bgcolor 0 -bgthresh 0 -maxxangle 1.1\
  -maxyangle 1.1 maxzangle 0.5 -maxidev 40 -w 50 -h 50"
```

Merge all `*.vec` files into a single `samples.vec` file:
```
python ./tools/mergevec.py -v samples/ -o samples.vec
```

## Training 
There are two ways in OpenCV to train cascade classifier.
* `opencv_haartraining` 
* `opencv_traincascade` - Newer version. Supports both Haar and LBP (Local Binary Patterns) 

To begin training using `opencv_traincascade`:
```
opencv_traincascade -data stage_outputs -vec samples.vec -bg negatives.txt\
  -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 390\
  -numNeg 600 -w 50 -h 50 -mode ALL -precalcValBufSize 8192\
  -precalcIdxBufSize 8192
```

Something important to note is that
> vec-file has to contain `>= [numPos + (numStages - 1) * (1 - minHitRate) * numPos] + S`, where `S` is a count of samples from vec-file that can be recognized as background right away

`numPos` and `numNeg` are the number of positive and negative samples we use in training for every classifier stage. Therefore, `numPos` should be relatively less than our total number of positive samples, taking into consideration the number of stages we'll be running. 

Each row of the training output for each stage represents a feature that's being trained. HR stands for Hit Ratio and FA stands for False Alarm. Note that if a training stage only has a few features (e.g. N = 1 ~ 3), that can suggest that the training data you used was not optimized.

## Testing Cascade
To test how well our cascade performs, run the `classifier.py` script. 

```
usage: classifier.py [-h] [-s] [-c] [-i] [-d] [-v] [-w] [-f] [-o] [-z] [-t]

Cascade Classifier

optional arguments:
  -h, --help    show this help message and exit
  -s, --save    specify output name
  -c, --cas     specify specific trained cascade
  -i, --img     specify image to be classified
  -d, --dir     specify directory of images to be classified
  -v, --vid     specify video to be classified
  -w, --cam     enable camera access for classification
  -f, --fps     enable frames text
  -o, --circle  enable circle detection
  -z, --scale   decrease video scale by scale factor
  -t, --track   select tracking algorithm [KCF, CSRT, MEDIANFLOW]
```

For example, to run a video through the classifier and save the output:
```
./classifier.py -v ~/video_input.MOV -s ~/video_output
```

## Video Conversions
The `classifier.py` automatically saves output videos as `*.avi` (fourcc: XVID). If you need other video types, this can be done very easily with `ffmpeg`. There are way more command arguments, especially if you want to consider encoding and compression types. The following command below converts the `*.avi` to `*.mp4` and compresses it.

```
ffmpeg -i video_input.avi -vcodec libx264 -crf 30 video_output.mp4
```

## Acknowledgements
For releasing their tools and notes under MIT license.
* [Naotoshi Seo](https://github.com/sonots) - `createsamples.pl`
* [Blake Wulfe](https://github.com/wulfebw) - `mergevec.py`
* [Thorsten Ball](https://github.com/mrnugget)

## References
* [OpenCV - Cascade Classifier Training](https://docs.opencv.org/master/dc/d88/tutorial_traincascade.html)
* [OpenCV - Face Detection using Haar Cascades](https://docs.opencv.org/master/d2/d99/tutorial_js_face_detection.html)
* [Naotoshi Seo - Tutorial: OpenCV haartraining](http://note.sonots.com/SciSoftware/haartraining.html)
* [Thorsten Ball - Train your own OpenCV Haar Classifier](https://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html)
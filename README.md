# Training, Classifying (Haar Cascades), and Tracking
Training our own Haar Cascade in OpenCV with Python. A cascade classifier has multiple stages of filtration. Using a sliding window approach, the image region within the window goes through the cascade. Can easily test accuracy of cascade with `classifier.py` script, which takes single images, directory of images, videos, and camera inputs. However, we want to also track our ROI (region of interest). This is because detection (through cascades, etc) is in general, more consuming and computationally taxing. Tracking algorithms are generally considered less taxing, because you know a lot about the apperance of the object in the ROI already. Thus, in the next frame, you use the information from the previous frame to predict and localize where the object is in the frames after.

There are many different types of tracking algorithms that are available through `opencv_contrib/tracking`, such as KCF, MOSSE, TLD, CSRT, etc. Here's a [good video](https://www.youtube.com/watch?v=61QjSz-oLr8) that demonstrates some of these tracking algorithms. Depending on your use case, the one you choose will differ.

## Jump to Section

* [Environment Setup](#environment-setup)
* [Image Scraping](#image-scraping)
* [Postive & Negative Image Sets](#positive-&-negative-image-sets)
* [Positive Samples Image Augmentation](#positive-samples-image-augmentation)
* [Training](#training)
* [Testing Cascade](#testing-cascade)
* [Video Conversions](#video-conversions)
* [Acknowledgements](#acknowledgements)
* [References](#references)

## Environment Setup
* Ubuntu 18.04; 20.04
* OpenCV 3.x.x (for running cascade training functions; built from source)
* OpenCV 4.4.0 (for running tracking algorithms)
* OpenCV Contrib (branch parallel with OpenCV 4.4.0)

As a clarification, the listed Ubuntu variants are for quick easy installs, depending on whether you're using `apt` or `pip`. To specifically get the OpenCV version you want, you will need to build from source (especially when you want to downgrade packages). I mainly used a Python virtual environment `venv` for package management. You can build and install OpenCV from source in the virtual environment (especially if you want a specific development branch or full control of compile options), or you can use `pip` locally in the `venv`. Packages included are shown in the `requirements.txt` file for reproducing the specific environment.

The project directory tree will look similar to the following below, and might change depending on the arguments passed to the scripts.

```
.
├── classifier.py
├── bin
│   └── createsamples.pl
├── negative_images
│   └── *.jpg / *.png
├── positive_images
│   └── *.jpg / *.png
├── negatives.txt
├── positives.txt
├── requirements.txt
├── samples
│   └── *.vec
├── stage_outputs
│   ├── cascade.xml
│   ├── params.xml
│   └── stage*.xml
├── tools
└── venv
```

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
We need to create a whole bunch of image samples, and we'll be using `OpenCV 3.x.x` to augment these images. [These tools / functionalities were disabled during legacy C API](https://github.com/opencv/opencv/issues/13231#issuecomment-440577461), so we'll need to first be on a downgraded version of OpenCV, and once we have our trained cascade model, we can upgrade back to `4.x.x`. As mentioned in the link earlier, most modern approaches use deep learning approaches. However having used Cascades, they still their applications! Anyways, to create a training set as a collection of PNG images:

```
opencv_createsamples -img ~/opencv-cascade-tracker/positive_images/img1.png -bg ~/opencv-cascade-tracker/negatives.txt -info ~/opencv-cascade-tracker/annotations/annotations.lst -pngoutput -maxxangle 0.1 -maxyangle 0.1 -maxzangle 0.1
```

But we need a whole bunch of these. To augment a set of positive samples with negative samples, let's run the perl script that Naotoshi Seo wrote:
```
perl bin/createsamples.pl positives.txt negatives.txt samples 1500\
  "opencv_createsamples -bgcolor 0 -bgthresh 0 -maxxangle 1.1\
  -maxyangle 1.1 maxzangle 0.5 -maxidev 40 -w 50 -h 50"
```

Merge all `*.vec` files into a single `samples.vec` file:
```
python ./tools/mergevec.py -v samples/ -o samples.vec
```

**Errors**: If you run into the following error when running `mergevec.py`:
```
Traceback (most recent call last):
  File "./tools/mergevec.py", line 170, in <module>
    merge_vec_files(vec_directory, output_filename)
  File "./tools/mergevec.py", line 133, in merge_vec_files
    val = struct.unpack('<iihh', content[:12])
struct.error: unpack requires a string argument of length 12
```

You need to remove all `*.vec` files with size 0 in your `samples` directory. Simply `cd samples` into the directory and double check with `ls -l -S` for file sizes, and run:

```
find . -type f -empty -delete
```

 Note: others have said that using artifical data vectors is not the best way to train a classifier. Personally, I have used this method and it worked fine for my use cases. However, you may approach this idea with a grain of salt and skip this step. 

## Training 
There are two ways in OpenCV to train cascade classifier.
* `opencv_haartraining` 
* `opencv_traincascade` - Newer version. Supports both Haar and LBP (Local Binary Patterns) 

These were the parameters I used for my initial cascade training. Later, we can introduce a larger dataset. To begin training using `opencv_traincascade`:

```
opencv_traincascade -data stage_outputs -vec samples.vec -bg negatives.txt\
  -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 390\
  -numNeg 600 -w 50 -h 50 -mode ALL -precalcValBufSize 8192\
  -precalcIdxBufSize 8192
```

The first cascade worked relatively well. However, performance suffered in different lighting conditions. As a result, I trained a second cascade with a larger dataset that included different lighting conditions..

```
opencv_traincascade -data stage_outputs -vec samples.vec -bg negatives.txt\
  -numStages 22 -minHitRate 0.993 -maxFalseAlarmRate 0.5 -numPos 1960\
  -numNeg 1000 -w 50 -h 50 -mode ALL -precalcValBufSize 16384\
  -precalcIdxBufSize 16384
```

Parameters for tuning `opencv_traincascade` are available in the [documentation](https://docs.opencv.org/4.4.0/dc/d88/tutorial_traincascade.html). `precalcValBufSize` and `precalcIdxBufSize` are buffer sizes. Currently set to 8192 Mb. If you have available memory, tune this parameter as training will be faster.

Something important to note is that
> vec-file has to contain `>= [numPos + (numStages - 1) * (1 - minHitRate) * numPos] + S`, where `S` is a count of samples from vec-file that can be recognized as background right away

`numPos` and `numNeg` are the number of positive and negative samples we use in training for every classifier stage. Therefore, `numPos` should be relatively less than our total number of positive samples, taking into consideration the number of stages we'll be running. 

```
===== TRAINING 0-stage =====
<BEGIN
POS count : consumed   1960 : 1960
NEG count : acceptanceRatio    1000 : 1
Precalculation time: 106
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1|    0.568|
+----+---------+---------+
|   5|  0.99949|    0.211|
+----+---------+---------+
END>
Training until now has taken 0 days 1 hours 3 minutes 47 seconds.
```

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
  -f, --fps     enable frames text (TODO)
  -o, --circle  enable circle detection
  -z, --scale   decrease video scale by scale factor
  -t, --track   select tracking algorithm [KCF, CSRT, MEDIANFLOW]
```

When testing a tracking algorithm, **pass the scale parameter**. For example, to run a video through the classifier and save the output:
```
./classifier.py -v ~/video_input.MOV -s ~/video_output -z 2 -t KCF
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
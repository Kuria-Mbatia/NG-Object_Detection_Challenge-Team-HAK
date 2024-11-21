#!/usr/bin/env python

import os
import sys
import tensorflow as tf
import argparse 
import json
import csv
from pathlib import Path

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

output_dir = os.path.join(ROOT_DIR,"results")


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib

# from mrcnn import utils
# from mrcnn import visualize
# from mrcnn.visualize import display_images
# from mrcnn.model import log


from samples.arms import unlabeled_arms

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Deploy trained Mask R-CNN model on a set of images.')
parser.add_argument('--dataset', required=False,
                    metavar="/path/to/image/dataset/",
                    help="Absolute path to directory of the images to segment and classify")
parser.add_argument('--weights', required=True,
                    metavar="/path/to/weights.h5",
                    help="Absolute path to weights .h5 file")
parser.add_argument('--output-dir', required=False,
                    metavar="/path/to/output/destination.h5",
                    help="Absolute path to directory where results will be output"
                    )

args = parser.parse_args()


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

if args.output_dir:

    output_dir = args.output_dir


output_dir = Path(output_dir)

if not os.path.exists(output_dir):

    os.makedirs(output_dir)


## Path to trained weights
# ARMS_WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs")
# ARMS_WEIGHTS_PATH = os.path.join(ARMS_WEIGHTS_PATH, "arms20230913T0025")
# ARMS_WEIGHTS_PATH = os.path.join(ARMS_WEIGHTS_PATH, "mask_rcnn_arms_0001.h5")
ARMS_WEIGHTS_PATH = args.weights


config = unlabeled_arms.ArmsConfig()
# ARMS_DIR = os.path.join(ROOT_DIR, "datasets")
# ARMS_DIR = os.path.join(ARMS_DIR, "arms")
ARMS_DIR = args.dataset

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Load validation dataset
dataset = unlabeled_arms.ArmsDataset()
dataset.load_images(ARMS_DIR) # new function in ArmsDataset class


# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
print(dataset.image_ids)


# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
#DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0 

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
    
# Set path to weights file
weights_path = ARMS_WEIGHTS_PATH

# Or, load the last model you trained
#weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


# iterating over datasets (directories in input area)
# for image_id in dataset.image_ids:
for image_id in dataset.image_ids[0:2]:
    
    # load image from modellib
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    
    # get image info
    info = dataset.image_info[image_id]
    
    # print some stuff
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))
    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    r = results[0]

    print(r['scores'])

    #### code from James ####

    original_stdout = sys.stdout # Save a reference to the original standard output

    reshapedMasks = r["masks"].reshape(r["masks"].shape[0], -1)

    # creating dataset-specific output dir
    curr_output_dir = Path(os.path.join(output_dir, info["id"]))
    os.makedirs(curr_output_dir)

    # capturing standard output
    std_out_filename = info["id"] + "_image_info.txt"
    std_out_path = Path(os.path.join(curr_output_dir, std_out_filename))
    
    with open(std_out_path, 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print(r)
        sys.stdout = original_stdout
    
    # creating csv
    csv_out_filename = info["id"] + "_flattenedWhole.csv"
    csv_out_path = Path(os.path.join(curr_output_dir, csv_out_filename))

    with open(csv_out_path,"w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(reshapedMasks)


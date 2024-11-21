"""
Mask R-CNN
Train on ARMS dataset using more than one class label.
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 labeled_arms.py train --dataset=/path/to/arms/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 labeled_arms.py train --dataset=/path/to/arms/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 labeled_arms.py train --dataset=/path/to/arms/dataset --weights=imagenet

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import multiprocessing
import tensorflow as tf
import keras

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to arms dataset
DATASET_PATH = os.path.join(ROOT_DIR, "datasets")
ARMS_DATASET_PATH = os.path.join(DATASET_PATH, "arms")

############################################################
#  Configurations
############################################################


class ArmsConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "arms"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 40 + 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    #DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_MIN_CONFIDENCE = 0.7


############################################################
#  Dataset
############################################################

class ArmsDataset(utils.Dataset):

    def load_arms(self, dataset_dir, subset):
        """Load a subset of the ARMS dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        # First argument is for dataset source name.
        self.add_class("arms", 1, "redPatch")
        self.add_class("arms", 2, "redBlob")
        self.add_class("arms", 3, "pinkPatch")
        self.add_class("arms", 4, "pinkBlob")
        self.add_class("arms", 5, "greenPatch")
        self.add_class("arms", 6, "greenBlob")
        self.add_class("arms", 7, "orangePatch")
        self.add_class("arms", 8, "orangeBlob")
        self.add_class("arms", 9, "redOrangeDot")
        self.add_class("arms", 10, "yellowPatch")
        self.add_class("arms", 11, "yellowBlob")
        self.add_class("arms", 12, "lightBrownPatch")
        self.add_class("arms", 13, "lightBrownBlob")
        self.add_class("arms", 14, "darkBrownPatch")
        self.add_class("arms", 15, "darkBrownBlob")
        self.add_class("arms", 16, "whitePatch")
        self.add_class("arms", 17, "whiteBlob")
        self.add_class("arms", 18, "whiteDot")
        self.add_class("arms", 19, "creamPatch")
        self.add_class("arms", 20, "creamBlob")
        self.add_class("arms", 21, "creamDot")
        self.add_class("arms", 22, "creamFuzz")
        self.add_class("arms", 23, "purplePatch")
        self.add_class("arms", 24, "purpleDot")
        self.add_class("arms", 25, "darkPatch")
        self.add_class("arms", 26, "darkBlob")
        self.add_class("arms", 27, "worm")
        self.add_class("arms", 28, "tube")
        self.add_class("arms", 29, "grassyPatch")
        self.add_class("arms", 30, "plant")
        self.add_class("arms", 31, "crescentPlant")
        self.add_class("arms", 32, "smallShell")
        self.add_class("arms", 33, "largeShell")
        self.add_class("arms", 34, "crab")
        self.add_class("arms", 35, "urchin")
        self.add_class("arms", 36, "branches")
        self.add_class("arms", 37, "net")
        self.add_class("arms", 38, "slug")
        self.add_class("arms", 39, "seaStar")
        self.add_class("arms", 40, "shrimp")

        class_dict = {
            "redPatch": 1,
            "redBlob": 2,
            "pinkPatch": 3,
            "pinkBlob": 4,
            "greenPatch": 5,
            "greenBlob": 6,
            "orangePatch": 7,
            "orangeBlob": 8,
            "redOrangeDot": 9,
            "yellowPatch": 10,
            "yellowBlob": 11,
            "lightBrownPatch": 12,
            "lightBrownBlob": 13,
            "darkBrownPatch": 14,
            "darkBrownBlob": 15,
            "whitePatch": 16,
            "whiteBlob": 17,
            "whiteDot": 18,
            "creamPatch": 19,
            "creamBlob": 20,
            "creamDot": 21,
            "creamFuzz": 22,
            "purplePatch": 23,
            "purpleDot": 24,
            "darkPatch": 25,
            "darkBlob": 26,
            "worm": 27,
            "tube": 28,
            "grassyPatch": 29,
            "plant": 30,
            "crescentPlant": 31,
            "smallShell": 32,
            "largeShell": 33,
            "crab": 34,
            "urchin": 35,
            "branches": 36,
            "net": 37,
            "slug": 38,
            "seaStar": 39,
            "shrimp": 40
        }

        # Train or validation dataset?
        assert subset in ["train", "val", "all"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # Custom JSON format in the form:
        # { 
        #   "originalJpeg": "PLATE_GUA-02_2014_B_011.JPG",
        #   "rotationDegrees": 90,
        #   "individuals":
        #   [
        #       {
        #           "label": "creamDot",
        #           "rgb": [238, 224, 156],
        #           "xs": [1582,1573,1552,1537,1522,1530,1552,1567],
        #           "ys": [2605,2626,2635,2620,2602,2583,2581,2594]
        #       },
        #   ]
        # }
        # We mostly care about the x and y coordinates of each region

        # each image has a directory in dataset folder
        # read json txt file for each image
        image_folders = os.listdir(dataset_dir)
        annotations = []
        image_names = []
        for folder in image_folders:
            current_path = os.path.join(dataset_dir, folder)
            annotations.append(json.load(open(os.path.join(current_path, folder + "_SUMMARY.txt"))))
            image_names.append(folder)

        # Add images
        for i in range(len(annotations)):
            a = annotations[i]

            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance.
            polygons = [individual for individual in a["individuals"]]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, image_names[i])
            image_path = os.path.join(image_path, image_names[i] + ".JPG")
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            #print(image_names[i])
            #print(a["rotationDegrees"])
            #print(width)
            #print(height)

            class_ids = []
            for polygon in polygons:
                class_ids.append(class_dict[polygon["label"]])

            # correct polygon orientation
            #if a["rotationDegrees"] == 0:
            if a["rotationDegrees"] == 90:
                for polygon in polygons:
                    temp = polygon['xs']
                    polygon['xs'] = polygon['ys']
                    polygon['ys'] = temp
                    for x in range(len(polygon['xs'])):
                        polygon['xs'][x] = (width - 1) - polygon['xs'][x]
            # elif a["rotationDegrees"] == 180:
            elif a["rotationDegrees"] == 270:
                for polygon in polygons:
                    temp = polygon['xs']
                    polygon['xs'] = polygon['ys']
                    polygon['ys'] = temp
                    for y in range(len(polygon['ys'])):
                        polygon['ys'][y] = (height - 1) - polygon['ys'][y]

            self.add_image(
                "arms",
                image_id=image_names[i],  # use folder name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                class_ids=class_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a arms dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "arms":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['ys'], p['xs'])
            #rr, cc = skimage.draw.polygon(p['xs'], p['ys'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance.
        class_ids = image_info["class_ids"]
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "arms":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ArmsDataset()
    dataset_train.load_arms(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ArmsDataset()
    dataset_val.load_arms(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

def evaluate(model):
    """Evaluate the model on validation set."""
    # Validation dataset
    dataset_val = ArmsDataset()
    dataset_val.load_arms(args.dataset, "val")
    dataset_val.prepare()

    # Data generators
    val_generator = modellib.data_generator(dataset_val, model.config, shuffle=True,
                                   batch_size=model.config.BATCH_SIZE)

    model.compile(config.LEARNING_RATE, model.config.LEARNING_MOMENTUM)

    # Work-around for Windows: Keras fails on Windows when using
    # multiprocessing workers. See discussion here:
    # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
    if os.name is 'nt':
        workers = 0
    else:
        workers = multiprocessing.cpu_count()

    losses = model.keras_model.evaluate_generator(
        val_generator, 
        steps=model.config.VALIDATION_STEPS,
        workers=workers, 
        use_multiprocessing=True 
    )
    print(model.keras_model.metrics_names)
    print(losses)

def cross_validation(model, num_folds):
    pass

def detect(model):
    pass

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to segment ARMS plates.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/arms/dataset/",
                        help='Directory of the ARMS dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ArmsConfig()
    else:
        class InferenceConfig(ArmsConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train" or args.command == "val":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "val":
        evaluate(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))

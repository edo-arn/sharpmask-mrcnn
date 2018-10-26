import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import model
from mrcnn import config
import tensorflow as tf
from keras import backend as KB


class CocoConfig(config.Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    NAME = "coco"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 80  # COCO has 80 classes + bg

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


cfg = CocoConfig()
mrcnn = model.MaskRCNN(mode='training', model_dir=MODEL_DIR, config=cfg)

#writer = tf.summary.FileWriter('../logs/tboard')
#writer.add_graph(KB.get_session().graph)
#writer.flush()






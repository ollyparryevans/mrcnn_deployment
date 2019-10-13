import unittest
import os
import tensorflow as tf
import random
import numpy as np
import pickle
import skimage.io

from mrcnn import utils, visualize
from mrcnn.model import MaskRCNN, load_image_gt, mold_image
from samples.shapes.shapes import ShapesDataset, ShapesConfig
from samples.coco.coco import CocoConfig, CocoDataset

random.seed(42)

sess = tf.InteractiveSession()

ROOT_DIR = os.path.abspath("../")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

model = MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

model.load_weights(COCO_MODEL_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])


# testing_data = CocoDataset().auto_download(os.path.join(ROOT_DIR, "coco_dataset"), "minival", "2014")
testing_data = CocoDataset()
testing_data.load_coco(os.path.join(ROOT_DIR, "coco_dataset"), "minival", "2014")
testing_data.prepare()

image_ids = np.random.choice(testing_data.image_ids, 5)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(testing_data, config,
                                                                     image_id, use_mini_mask=False)
    molded_images = np.expand_dims(mold_image(image, config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))


class TestPredictions(unittest.TestCase):

    def test_known_prediction(self):
        results = model.detect([testing_data.load_image(random.choice(testing_data.image_ids))])[0]
        with open(os.path.join(RESULTS_DIR, 'test_results.pickle'), 'rb') as handle:
            old_results = pickle.load(handle)

        for k, v in results.items():
            self.assertTrue(np.array_equal(old_results[k], v))

    def test_concept_drift(self):
        image_ids = np.random.choice(testing_data.image_ids, 5)
        APs = []
        for image_id in image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(testing_data, config,
                                                                             image_id, use_mini_mask=False)
            molded_images = np.expand_dims(mold_image(image, config), 0)
            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]
            # Compute AP
            AP, precisions, recalls, overlaps = \
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)

        self.assertGreaterEqual(np.mean(APs), 0.8)

# if __name__ == '__main__':
#     unittest.main()

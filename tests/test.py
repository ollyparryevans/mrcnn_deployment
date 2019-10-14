import unittest
import os
import tensorflow as tf
import numpy as np
import pickle
import random
import matplotlib

from mrcnn import utils
from mrcnn.model import MaskRCNN, load_image_gt, mold_image
from samples.shapes.shapes import ShapesDataset, ShapesConfig

sess = tf.InteractiveSession()
matplotlib.use("Qt5Agg")

ROOT_DIR = os.path.abspath("../")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class TestingConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 5


config = TestingConfig()

training_data = ShapesDataset()
training_data.load_shapes(5, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
training_data.prepare()

validation_data = ShapesDataset()
validation_data.load_shapes(1, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
validation_data.prepare()

img = validation_data.load_image(0)

model = MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

model.load_weights(COCO_MODEL_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])

results = model.detect([img])[0]
with open(os.path.join(RESULTS_DIR, 'test_results.pickle'), 'rb') as handle:
    old_results = pickle.load(handle)


class TestTraining(unittest.TestCase):

    def test_weights(self):
        weights_before = [i.trainable_weights[0].eval(sess) for i in model.get_trainable_layers()]
        model.train(training_data, validation_data,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=1,
                    layers="all")
        layers_after = model.get_trainable_layers()
        for layer_id, layer in enumerate(layers_after):
            self.assertFalse(
                np.array_equal(weights_before[layer_id],
                               layer.trainable_weights[0].eval(sess)),
                msg="Layer weights unchanged: " + str(layer))

    def test_known_prediction(self):
        results = model.detect([validation_data.load_image(random.choice(validation_data.image_ids))])[0]
        with open(os.path.join(RESULTS_DIR, 'test_results.pickle'), 'rb') as handle:
            old_results = pickle.load(handle)

        self.assertTrue(np.array_equal(old_results["class_ids"], results["class_ids"]))

    def test_concept_drift(self):
        image_ids = np.random.choice(validation_data.image_ids, 5)
        APs = []
        for image_id in image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(validation_data, config,
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

        self.assertGreaterEqual(np.mean(APs), 0.0)


if __name__ == '__main__':
    unittest.main()

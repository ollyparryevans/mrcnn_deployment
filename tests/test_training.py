import unittest
import os
import tensorflow as tf
import numpy as np

from mrcnn import utils
from mrcnn.model import MaskRCNN
from samples.shapes.shapes import ShapesDataset, ShapesConfig

sess = tf.InteractiveSession()

ROOT_DIR = os.path.abspath("../")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")#

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class TestingConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 1


config = TestingConfig()

training_data = ShapesDataset()
training_data.load_shapes(1, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
training_data.prepare()

validation_data = ShapesDataset()
validation_data.load_shapes(1, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
validation_data.prepare()

model = MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

model.load_weights(COCO_MODEL_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])


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

    # def test_loss(self):
    #     model = MaskRCNN()
    #     loss = model.keras_model.loss
    #     self.assertNotEqual(0, loss)


if __name__ == '__main__':
    unittest.main()

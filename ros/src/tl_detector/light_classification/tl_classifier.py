from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import os
import cv2
import rospy
import yaml

MAX_IMAGE_WIDTH = 300
MAX_IMAGE_HEIGHT = 300
SAVE_IMAGE = False
SAVE_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../../../../test_images/simulator/'


class TLClassifier(object):
    def __init__(self):
        self.nn_graph = None
        self.session = None
        self.classes = {1: TrafficLight.RED,
                        2: TrafficLight.YELLOW,
                        3: TrafficLight.GREEN,
                        4: TrafficLight.UNKNOWN}

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.load_model(self.get_path())
        self.count = 0

    def load_model(self, model_path):
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        self.nn_graph = tf.Graph()
        with tf.Session(graph=self.nn_graph, config=config) as sess:
            self.session = sess
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

    def get_path(self):
        return os.path.dirname(os.path.realpath(__file__)) + self.config['detection_model']

    def preprocess_image(self, image):
        image = cv2.resize(image, (MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def predict(self, img, min_score=0.5):
        img = self.preprocess_image(img)
        img_tensor = self.nn_graph.get_tensor_by_name('image_tensor:0')
        det_boxes = self.nn_graph.get_tensor_by_name('detection_boxes:0')
        det_scores = self.nn_graph.get_tensor_by_name('detection_scores:0')
        det_classes = self.nn_graph.get_tensor_by_name('detection_classes:0')

        boxes, scores, classes = self.session.run([det_boxes, det_scores, det_classes], feed_dict={img_tensor: np.expand_dims(img, axis=0)})
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        boxes = np.squeeze(boxes)

        for i, _ in enumerate(boxes):
            if scores[i] > min_score:
                light_class = self.classes[classes[i]]
                if SAVE_IMAGE:
                    self.save_image(img, light_class)
                
                rospy.logdebug("Traffic Light Class detected: %d", light_class)
                return light_class, scores[i]
            else:
                
                if SAVE_IMAGE:
                    self.save_image(img, TrafficLight.UNKNOWN)

        return None, None

    def get_classification(self, image):
        class_index, prob = self.predict(image)

        if class_index is not None:
            rospy.logdebug("class: %d, probability: %f", class_index, prob)

        return class_index

    def save_image(self, image, light_class):
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(SAVE_PATH, "image_%04i_%d.jpg" % (self.count, light_class)), bgr_image)
            self.count += 1

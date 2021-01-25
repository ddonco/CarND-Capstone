#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight, Waypoint
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import math
import sys
import tf
from timeit import default_timer as timer
import yaml

FREQUENCY = 10.
MAX_DISTANCE = 100000
STATE_COUNT_THRESHOLD = 3


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        rospy.logwarn("### Traffic Light Detector Initialization ....")

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.image_processing_time = 0

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        if self.image_processing_time > 0:
            self.image_processing_time -= self.hertz_to_seconds(FREQUENCY)
            return

        self.has_image = True
        self.camera_image = msg
        start_detection = timer()
        light_wp, state = self.process_traffic_lights()
        end_detection = timer()
        self.image_processing_time = end_detection - start_detection

        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state

            if state == TrafficLight.RED or state == TrafficLight.YELLOW:
                light_wp = light_wp
            else:
                light_wp = -1

            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def hertz_to_seconds(self, hertz):
        return hertz / 60

    def get_light_state(self, light):
        if (not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification
        predicted = self.light_classifier.get_classification(cv_image)

        rospy.logdebug("traffic light state: %d", light.state)

        return predicted

    def get_nearest_index(self, pose, positions):
        index = -1
        min_distance = MAX_DISTANCE

        for i in range(len(positions)):
            dist = self.distance_between_pos(pose, positions[i].pose.pose.position)
            if dist < min_distance:
                index = i
                min_distance = dist

        return index

    def get_nearest_waypoint(self, pose):
        return self.get_nearest_index(pose, self.waypoints.waypoints)

    def get_nearest_stop_line(self, pose):
        return self.get_nearest_index(pose, self.get_stop_lines())

    def get_stop_lines(self):
        stop_lines = []
        for light_position in self.config['stop_line_positions']:
            point = Waypoint()
            point.pose.pose.position.x = light_position[0]
            point.pose.pose.position.y = light_position[1]
            point.pose.pose.position.z = 0.0
            stop_lines.append(point)
        return stop_lines

    def get_nearest_light(self, pose):
        return self.get_nearest_index(pose, self.lights)

    def distance_between_pos(self, pos_a, pos_b):
        return math.sqrt((pos_a.x - pos_b.x) ** 2 + (pos_a.y - pos_b.y) ** 2 + (pos_a.z - pos_b.z) ** 2)

    def process_traffic_lights(self):
        light = None
        distance_tolerance = 300
        stop_line_position = None

        if self.pose:
            car_index = self.get_nearest_waypoint(self.pose.pose.position)
            car_position = self.waypoints.waypoints[car_index].pose.pose.position

            light_index = self.get_nearest_light(car_position)
            if light_index != -1:
                light_waypoint_index = self.get_nearest_waypoint(self.lights[light_index].pose.pose.position)
                light_position = self.waypoints.waypoints[light_waypoint_index].pose.pose.position

                if light_waypoint_index > car_index:
                    distance_to_traffic_light = self.distance_between_pos(car_position, light_position)
                    if distance_to_traffic_light < distance_tolerance:
                        light = self.lights[light_index]
                        stop_line_index = self.get_nearest_stop_line(light_position)
                        stop_line_position = self.get_stop_lines()[stop_line_index].pose.pose
                        stop_line_waypoint = self.get_nearest_waypoint(stop_line_position.position)

        if light and stop_line_position:
            state = self.get_light_state(light)
            return stop_line_waypoint, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

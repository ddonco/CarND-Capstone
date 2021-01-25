#!/usr/bin/env python

import rospy
import sys
from geometry_msgs.msg import TwistStamped, PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Bool, Int32, Float32

import copy
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100
FREQUENCY = 10.
MAX_DISTANCE = 100000
DISTANCE_BUFFER = 1.


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.logwarn("### WaypointUpdater Initialization ....")

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/obstacle_waypoints', PoseStamped, self.obstacle_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.final_waypoints = []

        self.num_waypoints = None
        self.current_lane = None
        self.current_position = None
        self.current_velocity = None
        self.current_state = 0  # 0 = deceleration, 1 = acceleration
        self.changed_state = True
        self.next_stopline_index = -1

        self.velocity_max = rospy.get_param('/waypoint_loader/velocity') / 3.6
        self.acceleration_limit = rospy.get_param('~accel_limit', 1.)
        self.deceleration_min_limit = min(1.0, -rospy.get_param('~decel_limit', -5.) / 2.)
        self.deceleration_max_limit = -rospy.get_param('~decel_limit', -5.)

        self.publish_loop()

    def pose_cb(self, msg):
        self.current_position = msg.pose.position

    def waypoints_cb(self, waypoints):
        self.current_lane = waypoints
        self.num_waypoints = len(self.current_lane.waypoints)

    def traffic_cb(self, msg):
        self.next_stopline_index = msg.data
        if self.next_stopline_index != -1 and self.num_waypoints is not None:
            self.next_stopline_index = (self.next_stopline_index - 5 + self.num_waypoints) % self.num_waypoints

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def positions_distance(self, pos_a, pos_b):
        return math.sqrt((pos_a.x - pos_b.x) ** 2 + (pos_a.y - pos_b.y) ** 2 + (pos_a.z - pos_b.z) ** 2)

    def check_waypoint_behind(self, wp_index):
        dx = self.current_position.x - self.current_lane.waypoints[wp_index].pose.pose.position.x
        dy = self.current_position.y - self.current_lane.waypoints[wp_index].pose.pose.position.y

        nx = self.current_lane.waypoints[(wp_index + 1) % self.num_waypoints].pose.pose.position.x - self.current_lane.waypoints[wp_index].pose.pose.position.x
        ny = self.current_lane.waypoints[(wp_index + 1) % self.num_waypoints].pose.pose.position.y - self.current_lane.waypoints[wp_index].pose.pose.position.y

        dp = dx * nx + dy * ny
        return dp > 0.

    def get_closest_waypoint_index(self):
        minimal_distance = MAX_DISTANCE
        waypoints = self.current_lane.waypoints
        wp_index = -1

        for i in range(self.num_waypoints):
            distance = self.positions_distance(self.current_position, waypoints[i].pose.pose.position)
            if distance < minimal_distance:
                minimal_distance = distance
                wp_index = i

        if self.check_waypoint_behind(wp_index):
            wp_index = (wp_index + 1) % self.num_waypoints

        return wp_index

    def accelerate_vehicle(self, lane, current_waypoint_index):
        current_velocity = self.current_velocity
        target_velocity = self.current_velocity
        acceleration = self.acceleration_limit

        count = 0
        while target_velocity < self.velocity_max or count < LOOKAHEAD_WPS:
            start_pos = self.current_position
            end_pos = self.current_lane.waypoints[(current_waypoint_index + count) % self.num_waypoints].pose.pose.position
            distance = self.positions_distance(start_pos, end_pos)
            target_velocity = math.sqrt(current_velocity ** 2. + 2. * acceleration * distance)

            if target_velocity > self.velocity_max:
                target_velocity = self.velocity_max

            current_waypoint = copy.deepcopy(self.current_lane.waypoints[(current_waypoint_index + count) % self.num_waypoints])
            current_waypoint.twist.twist.linear.x = target_velocity
            lane.waypoints.append(current_waypoint)
            count += 1

    def decelerate_vehicle(self, lane, current_waypoint_index):
        current_velocity = self.current_velocity
        target_velocity = self.current_velocity
        distance = self.positions_distance(self.current_position, self.current_lane.waypoints[self.next_stopline_index].pose.pose.position) - DISTANCE_BUFFER
        acceleration = current_velocity ** 2. / (2. * distance)

        count = 0
        while target_velocity > 0. or count < LOOKAHEAD_WPS:
            start_pos = self.current_position
            end_pos = self.current_lane.waypoints[(current_waypoint_index + count) % self.num_waypoints].pose.pose.position
            distance = self.positions_distance(start_pos, end_pos)
            target_velocity_exp = current_velocity ** 2. - 2. * acceleration * distance

            if target_velocity_exp <= 0:
                target_velocity = 0

            else:
                target_velocity = math.sqrt(target_velocity_exp)

            current_waypoint = copy.deepcopy(self.current_lane.waypoints[(current_waypoint_index + count) % self.num_waypoints])
            current_waypoint.twist.twist.linear.x = target_velocity
            lane.waypoints.append(current_waypoint)
            count += 1

    def maintain_state(self, lane, wp_index, current_velocity):
        count = 0
        while count < len(self.final_waypoints):

            if self.final_waypoints[count].pose.pose.position == self.current_lane.waypoints[wp_index].pose.pose.position:
                break
            count += 1

        for i in range(count, len(self.final_waypoints)):
            current_waypoint = copy.deepcopy(self.current_lane.waypoints[(wp_index + i - count) % self.num_waypoints])
            current_waypoint.twist.twist.linear.x = self.final_waypoints[i].twist.twist.linear.x
            lane.waypoints.append(current_waypoint)

        for i in range(len(lane.waypoints), LOOKAHEAD_WPS):
            current_waypoint = copy.deepcopy(self.current_lane.waypoints[(wp_index + i) % self.num_waypoints])
            current_waypoint.twist.twist.linear.x = current_velocity
            lane.waypoints.append(current_waypoint)

    def publish(self):
        if self.current_position is None or self.current_lane is None or self.current_velocity is None:
            return

        wp_index = self.get_closest_waypoint_index()
        lane = Lane()
        lane.header.stamp = rospy.Time.now()

        if self.current_state == 1:

            if self.next_stopline_index != -1:
                start_pos = self.current_position
                end_pos = self.current_lane.waypoints[self.next_stopline_index].pose.pose.position
                brake_dist = self.positions_distance(start_pos, end_pos) - DISTANCE_BUFFER

                min_brake_distance = 0.5 * self.current_velocity ** 2 / self.deceleration_max_limit
                max_brake_distance = 0.5 * self.current_velocity ** 2 / self.deceleration_min_limit

                if max_brake_distance >= brake_dist >= min_brake_distance:
                    self.changed_state = True
                    self.current_state = 0

        elif self.current_state == 0:

            if self.next_stopline_index == -1:
                self.changed_state = True
                self.current_state = 1

        if self.current_state == 0 and not self.changed_state:
            self.maintain_state(lane, wp_index, 0)

        elif self.current_state == 0 and self.changed_state:
            self.decelerate_vehicle(lane, wp_index)

        elif self.current_state == 1 and not self.changed_state:
            self.maintain_state(lane, wp_index, self.velocity_max)

        elif self.current_state == 1 and self.changed_state:
            self.accelerate_vehicle(lane, wp_index)

        self.changed_state = False
        self.final_waypoints = copy.deepcopy(lane.waypoints)
        self.final_waypoints_pub.publish(lane)

    def publish_loop(self):
        rate = rospy.Rate(FREQUENCY)
        while not rospy.is_shutdown():
            self.publish()
            rate.sleep()

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')

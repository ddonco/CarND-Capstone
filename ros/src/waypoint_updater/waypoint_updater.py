#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math
import tf

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

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
TARGET_SPEED_MPH = 10.0 # Target speed is 10 MPH
STOP_DISTANCE = 5.0 # Desired stopping distance is 5 meters
MAX_DECELERATION = 0.5 # Maximum vehicle deceleration is 0.5 m/s^2
DEBUG = True


def euclidian_distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb, queue_size=1)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)
        self.waypoints = None
        self.waypoint_red_light = None
        self.current_position = None

        rospy.spin()

    def pose_cb(self, msg):
        self.current_position = msg.pose
        
        if self.waypoints is not None:
            self.publish()

    def waypoints_cb(self, waypoints):
        if self.waypoints is not None:
            self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.waypoint_red_light = msg.data
        rospy.loginfo("Red Light Detected: {}".format(str(msg.data)))

        if self.waypoint_red_light > -1:
            self.publish()

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def nearest_waypoint(self, pose):
        nearest_dist = 10000
        nearest_waypoint = 0

        for i, waypoint in enumerate(self.waypoints):
            dist = euclidian_distance(pose.position, waypoint.pose.pose.position)

            if dist < nearest_dist:
                nearest_waypoint = i
                nearest_dist = dist
        
        return nearest_waypoint

    def next_waypoint(self, pose, waypoints):
        waypoint = self.nearest_waypoint(pose)

        mapx = waypoints[waypoint].pose.pose.position.x
        mapy = waypoints[waypoint].pose.pose.position.y

        heading = math.atan2((mapy - pose.position.y), (mapx - pose.position.x))

        _, _, yaw = tf.transformations.euler_from_quaternion((pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))
        
        angle = abs(yaw - heading)

        if angle > math.pi / 4:
            waypoint += 1

        return waypoint

    def decelerate(self, waypoints, waypoint_index_red_light):
        if len(waypoints) < 1:
            return []

        # first_waypoint = waypoints[0]
        last_waypoint = waypoints[waypoint_index_red_light]

        last_waypoint.twist.twist.linear.x = 0.
        # dist_total = self.distance(waypoints, first_waypoint.pose.pose.position, last_waypoint.pose.pose.position)
        # start_velocity = first_waypoint.twist.twist.linear.x

        velocity = 0.
        for i, waypoint in enumerate(waypoints):
            if i <= waypoint_index_red_light:
                dist = self.distance(waypoints, waypoint.pose.pose.position, last_waypoint.pose.pose.position)
                dist = max(0, dist - STOP_DISTANCE)
                velocity = math.sqrt(dist * MAX_DECELERATION * 2)

                if velocity < 1.:
                    velocity = 0.
                
            waypoint.twist.twist.linear.x = min(velocity, waypoint.twist.twist.linear.x)

    def publish(self):
        if self.current_position is not None:
            waypoint_index_next = self.next_waypoint(self.current_position, self.waypoints)
            waypoints_lookahead = self.waypoints[waypoint_index_next:waypoint_index_next+LOOKAHEAD_WPS]

            if self.waypoint_red_light is None or self.waypoint_red_light < 0:
                for i in range(len(waypoints_lookahead) - 1):
                    velocity = (TARGET_SPEED_MPH * 1609.34) / 3600 # 10 MPH in mps
                    self.set_waypoint_velocity(waypoints_lookahead, i, velocity)

            else:
                lookahead_redlight_index = max(0, self.waypoint_red_light - waypoint_index_next)
                waypoints_lookahead = self.decelerate(waypoints_lookahead, lookahead_redlight_index)

            lane = Lane()
            lane.header.stamp = rospy.Time(0)
            lane.header.frame_id = '/world'
            lane.waypoints = waypoints_lookahead

            self.final_waypoints_pub.publish(lane)

            if DEBUG:
                posx = self.waypoints[waypoint_index_next].pose.pose.position.x
                posy = self.waypoints[waypoint_index_next].pose.pose.position.y
                rospy.loginfo("Nearest waypoint (i, x, y): ({d}, {:.2f}, {:.2f})".format(waypoint_index_next, posx, posy))


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')

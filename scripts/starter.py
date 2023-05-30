#!/usr/bin/env python
# ----------------------------------------------------- #
# Project 2
# ----------------------------------------------------- #
# TODO: Fill in team information
# Team Robot:
# Group Member Name and ID:
# Group Member Name and ID:
# Group Member Name and ID:


######################################################################
# Import Libraries


import roslib; roslib.load_manifest('project2')
import rospy
import rospkg
import sys

from geometry_msgs.msg import Twist
from kobuki_msgs.msg import SensorState
from blobfinder.msg import MultiBlobInfo3D

from transform2d import transform2d_from_ros_transform, distance_2d, Transform2D

import tf
import math
import numpy as np


######################################################################
# Global parameters


# Image size
image_size = {'width': 640, 'height': 480} 

# control at 100Hz
CONTROL_PERIOD = rospy.Duration(0.01)

# minimum duration of safety stop (s)
STOP_DURATION = rospy.Duration(1.0)

# minimum distance of cones to make goal (m)
MIN_GOAL_DIST = 0.55

# maximum distance of cones to make goal (m)
MAX_GOAL_DIST = 1.25

# minimum number of pixels to identify cone (px)
MIN_CONE_AREA = 200
MIN_BALL_AREA = 100



######################################################################
# Helper functions


def clamp(x, lo, hi):
    """Returns lo, if x < lo; hi, if x > hi, or x otherwise"""
    return max(lo, min(x, hi))

def filter_vel(prev_vel, desired_vel, max_change):
    """Update prev_val towards desired_vel at a maximum rate of max_change

    This should return desired_vel if absolute difference from prev_vel is
    less than max_change, otherwise returns prev_vel +/- max_change,
    whichever is closer to desired_vel.
    """
    # TODO: writeme - you may find clamp() above helpful
    return desired_vel


def angle_from_vec(v):
  """Construct the angle between the vector v and the x-axis."""
  return np.arctan2(v[1], v[0])


def offset_vec(v, r):
    """Offset a given vector v by a fixed distance r."""
    return v * (1 + r / np.linalg.norm(v))


###################################################################### 
# Controller class


class Controller:
    """Class to handle our simple controller"""

    # initialize our controller
    def __init__(self):

        # initialize our ROS node
        rospy.init_node('starter')

        # Get goal information (change if you are using different cones)
        self.goal_cone_colors = ('green', 'yellow')

        # set up publisher for commanded velocity
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity',
                                           Twist, queue_size=10)

        # set up a TransformListener to get odometry information
        self.odom_listener = tf.TransformListener()
        
        # record whether we should stop for safety
        self.should_stop = 0
        self.time_of_stop = rospy.get_rostime() - STOP_DURATION

        # initialize a dictionary of two empty lists of cones. each
        # list should hold world-frame locations of cone in XY
        self.cone_locations = dict(yellow=[], green=[])

        # these bool values will be set to True when we get a new cone/ball message
        self.cones_updated = False
        self.ball_updated = False

        # No goal or ball yet
        self.goal_in_world_frame = None
        self.ball_in_world_frame = None

        # Start at 0 velocity
        self.prev_cmd_vel = Twist()

        # Motion smoothing
        # maximum change in linear velocity per 0.01s timestep
        MAX_LINEAR_CHANGE = 1000000 # TODO: set to reasonable value
        
        # maximum change in angular velocity per 0.01s timestep
        MAX_ANGULAR_CHANGE = 1000000 # TODO: set to reasonable value

        # set up our trivial 'state machine' controller
        rospy.Timer(CONTROL_PERIOD,
                    self.control_callback)
        
        # set up subscriber for sensor state for bumpers/cliffs
        rospy.Subscriber('/mobile_base/sensors/core',
                         SensorState, self.sensor_callback)

        # set up subscriber for yellow cones, green cones, and blue ball
        rospy.Subscriber('/blobfinder2/blobs3d', 
                         MultiBlobInfo3D, self.blobs3d_callback)

        ## INITIALIZE CUSTOM STATE VARIABLES HERE ##
        ## end __init__


    #################################################################
    # Sensor callback functions


    def sensor_callback(self, msg):
     """Called when sensor msgs received

     Just copy sensor readings to class member variables"""

        if msg.bumper & SensorState.BUMPER_LEFT:
            rospy.loginfo('***LEFT BUMPER***')
        if msg.bumper & SensorState.BUMPER_CENTRE:
            rospy.loginfo('***MIDDLE BUMPER***')
        if msg.bumper & SensorState.BUMPER_RIGHT:
            rospy.loginfo('***RIGHT BUMPER***')
        if msg.cliff:
            rospy.loginfo('***CLIFF***')

        if msg.bumper or msg.cliff:
            self.should_stop = True
            self.time_of_stop = rospy.get_rostime()
        else:
            self.should_stop = False


    def blobs3d_callback(self, msg):
        """Called when a blob message comes in

        Calls other methods to process blobs"""

        num = len(msg.color_blobs)
        for i in range(num):
            color_blob = msg.color_blobs[i]
            color = color_blob.color.data
            if color == 'red_tape':
                # Don't need to look at red tape for this lab.
                pass
            elif color == 'yellow_cone':
                self.cone_callback(color_blob, 'yellow')
            elif color == 'green_cone':
                self.cone_callback(color_blob, 'green')
            elif color == 'blue_ball':
                self.ball_callback(color_blob)


    def cone_callback(self, bloblist, color):
        """Called when a cone related blob message comes in

        Sets cones_updated to true if cones detected. Updates cone_locations
        by appopriate color"""

        T_world_from_robot = self.get_current_pose()
        if T_world_from_robot is None:
            rospy.logwarn('no xform yet in blobs3d_callback')
            return

        blob_locations = []

        for blob3d in bloblist.blobs:
            if blob3d.have_pos and blob3d.blob.area > MIN_CONE_AREA and blob3d.blob.cy > 240:
                blob_in_robot_frame = np.array([blob3d.position.z, -blob3d.position.x])
                blob_dir = blob_in_robot_frame / np.linalg.norm(blob_in_robot_frame)
                blob_in_robot_frame += blob_dir * 0.04 # offset radius
                blob_locations.append( T_world_from_robot * blob_in_robot_frame )

        self.cone_locations[color] = blob_locations
        self.cones_updated = True


    def ball_callback(self, bloblist):
        """Called when a ball blob message comes in

        Sets ball_updated to true if ball detected. Sets ball_in_world_frame
        value if the ball is detected."""

        T_world_from_robot = self.get_current_pose()
        if T_world_from_robot is None:
            rospy.logwarn('no xform yet in blobs3d_callback')
            return

        # Get max area blob above minmum threshold
        blob_location = None
        blob_area = -1
        for blob3d in bloblist.blobs:
            if blob3d.have_pos and blob3d.blob.area > MIN_BALL_AREA and blob3d.blob.cy > 240:
                if blob3d.blob.area > blob_area:
                    # Get position with calibrated corrections
                    blob_in_robot_frame = np.array([blob3d.position.z, -blob3d.position.x])
                    blob_dir = blob_in_robot_frame / np.linalg.norm(blob_in_robot_frame)
                    blob_in_robot_frame += blob_dir * 0.04 # offset radius
                    # Update area and location to track max
                    blob_location = T_world_from_robot * blob_in_robot_frame
                    blob_area = blob3d.blob.area

        if blob_location is not None:
            self.ball_updated = True
            self.ball_in_world_frame = Transform2D(blob_location[0], blob_location[1], 0)

    # get current pose from TransformListener
    def get_current_pose(self):

        try:

            ros_xform = self.odom_listener.lookupTransform(
                '/odom', '/base_footprint',
                rospy.Time(0))

        except tf.LookupException:

            return None

        return transform2d_from_ros_transform(ros_xform)


    def find_goal_location(self, cur_pose):
        """Function to find goal location from cones.

        Called when we need to find the goal. Retunrs goal in world coordinates"""

        # build a list of all cones by combining colors
        all_cones = []

        for color in ['yellow', 'green']:
            for world_xy in self.cone_locations[color]:
                all_cones.append( (color, world_xy) )

        rospy.loginfo(all_cones)
        # get the inverse transformation of cur pose to be able to map
        # points back into robot frame (so we can assess which cones
        # are left and right)
        T_robot_from_world = cur_pose.inverse()

        # build a list of all possible goals by investigating pairs of cones
        goal = None

        # for each cone
        for i in range(len(all_cones)):

            # get color, world pos, robot pos
            (ci, wxy_i) = all_cones[i]
            if ci != self.goal_cone_colors[0]:
                continue
            rxy_i = T_robot_from_world * wxy_i

            # for each other cone
            for j in range(len(all_cones)):
                if j == i:
                    continue

                # get color, world pos, robot pos
                (cj, wxy_j) = all_cones[j]
                if cj != self.goal_cone_colors[1]:
                    continue
                rxy_j = T_robot_from_world * wxy_j

                # get distance between cone pair
                dist_ij = distance_2d(wxy_i, wxy_j)
                
                # if in the range that seems reasonable for a goal:
                rospy.loginfo('Dists: ' + str([dist_ij, MIN_GOAL_DIST]))
                if dist_ij > MIN_GOAL_DIST and dist_ij < MAX_GOAL_DIST:

                    # get midpoint of cones
                    midpoint = 0.5 * (wxy_i + wxy_j)

                    # sort the cones left to right (we want left to
                    # have a larger Y coordinate in the robot frame)
                    if rxy_i[1] > rxy_j[1]:
                        lcolor, wxy_l, rcolor, wxy_r = ci, wxy_i, cj, wxy_j
                    else:
                        lcolor, wxy_l, rcolor, wxy_r = cj, wxy_j, ci, wxy_i

                    # the direction of the goal frame's local Y axis
                    # is obtained by subtracting the right cone from
                    # the left cone.
                    goal_ydir = (wxy_l - wxy_r) 

                    # the angle of rotation for the goal is obtained
                    # by the arctangent given here
                    goal_theta = math.atan2(-goal_ydir[0], goal_ydir[1])
                    
                    T_world_from_goal = Transform2D(midpoint[0], midpoint[1], 
                                                   goal_theta)

                    goal = T_world_from_goal
                    break # break from loop after we find goal
            # OK
            if goal is not None:
                break

        self.cones_updated = False

        return goal

    #################################################################
    # TODO Control functions -- for you to implement

    # Turn towards
    def turn_towards(self, point):
        cmd_vel = Twist()
        # TODO: task 1 - make the robot turn towards the specified point
        return cmd_vel

    # Drive towards
    def drive_towards(self, point):
        cmd_vel = Twist()
        # TODO: task 1 - make the robot turn towards the specified point
        return cmd_vel

    # called periodically to do top-level coordination of behaviors
    def control_callback(self, timer_event=None):

        # initialize vel to 0, 0
        cmd_vel = Twist()

        time_since_stop = rospy.get_rostime() - self.time_of_stop

        cur_pose = self.get_current_pose()

        if self.should_stop or time_since_stop < STOP_DURATION:
            
            rospy.loginfo('stopped')
            
        else: # not stopped for safety

            # TODO: task 1 - hard-code a commanded velocity and angular velocity
            # here to test & verify filter_vel, then remove when you are happy
            # with filtering behavior

            if self.cones_updated and cur_pose is not None:
                goal_location = self.find_goal_location(cur_pose)

                # now update self.goal_location
                if goal_location is not None:
                    self.goal_in_world_frame = goal_location
                    rospy.loginfo('updated current goal!')

            if self.goal_in_world_frame is None:

                rospy.loginfo('waiting for goal...')
                # For task 2, you might want to uncomment this
                # cmd_vel.angular.z = 1.0 # Rotate until we see goal

            else:

                T_world_from_robot = cur_pose
                T_world_from_goal = self.goal_in_world_frame
                T_world_from_ball = self.ball_in_world_frame

                dist_to_goal = distance_2d(T_world_from_robot.translation(),
                                           T_world_from_goal.translation())

                rospy.loginfo('goal has cones {}-{} and is at {} in world coords, {} meters away'.format(
                    self.goal_in_world_frame[1], self.goal_in_world_frame[2],
                    T_world_from_goal.translation(),
                    dist_to_goal))

                # TODO: task 2 - set cmd_vel to either turn towards a point
                #                or move towards the point

            # now filter large changes in velocity before commanding
            # robot - note we don't filter when stopped
            cmd_vel.linear.x = filter_vel(self.prev_cmd_vel.linear.x,
                                          cmd_vel.linear.x,
                                          MAX_LINEAR_CHANGE)

            cmd_vel.angular.z = filter_vel(self.prev_cmd_vel.angular.z,
                                           cmd_vel.angular.z,
                                           MAX_ANGULAR_CHANGE)

        self.cmd_vel_pub.publish(cmd_vel)
        self.prev_cmd_vel = cmd_vel


    #################################################################
    # ROS related functions 


    def run(self):
        """Called by main function below (after init)"""
        
        # timers and callbacks are already set up, so just spin
        rospy.spin()

        # if spin returns we were interrupted by Ctrl+C or shutdown
        rospy.loginfo('goodbye')


######################################################################
# Main function


if __name__ == '__main__':
    try:
        ctrl = Controller()
        ctrl.run()
    except rospy.ROSInterruptException:
        pass
    

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
from blobfinder2.msg import MultiBlobInfo3D

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
MAX_GOAL_DIST = 2.5

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
    abs_diff = abs(prev_vel - desired_vel)
    if abs_diff < max_change:
        return desired_vel
    else:
        prev_plus = abs(prev_vel + max_change - desired_vel)
        prev_minus = abs(prev_vel - max_change - desired_vel)
        if prev_plus < prev_minus:
            return prev_vel + max_change
        else:
            return prev_vel - max_change
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
        self.MAX_LINEAR_CHANGE = 0.01 # TODO: set to reasonable value
        
        # maximum change in angular velocity per 0.01s timestep
        self.MAX_ANGULAR_CHANGE = 0.005 # TODO: set to reasonable value
	#Task 1 variables
        self.start_pose = None
        self.target_position = None
	self.target_in_robot = None
        self.toDrive = False
	self.target_reached = False
        self.to_kick = False
        self.ball_visible = False
	self.start_kick = False
        self.time_start_kick = 0
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
            self.ball_visible = True
        else: 
            self.ball_visible = False

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
    def turn_to_target(self, point):
        cmd_vel = Twist()
        # TODO: task 1 - make the robot turn towards the specified point
        beta = angle_from_vec(point)
        if abs(beta) <= (math.pi/180):
            self.toDrive = True
        cmd_vel.angular.z = 0.5*beta + 0.3*np.sign(beta) # k was 0.1 
        cmd_vel.linear.x = 0

	return cmd_vel

    # Drive towards
    def drive_to_target(self, point):
        cmd_vel = Twist()
        beta = angle_from_vec(point)
        # TODO: task 1 - make the robot turn towards the specified point
	cmd_vel.linear.x = 0.4*point[0] + 0.4

        if point[0] > 0.1:
            cmd_vel.angular.z = 0.1*beta
        else:
            cmd_vel.angular.z = 0
	    cmd_vel.linear.x = 0
	    self.target_reached = True
        return cmd_vel

    
    def turn_to_ball(self, point):
        cmd_vel = Twist()
        beta = angle_from_vec(point)
        if abs(beta) <= (math.pi/180+0.15): 
            self.to_kick = True
        cmd_vel.angular.z = 0.3*beta + 0.3*np.sign(beta)
        cmd_vel.linear.x = 0
        return cmd_vel

    def hit_ball(self, point):	
    	cmd_vel = Twist() 
        beta = angle_from_vec(point)
        cmd_vel.linear.x = np.abs(10*point[0] + 10)
        if point[0] > 0.1: 
            cmd_vel.angular.z = -0.1*abs(beta) 
       	    #cmd_vel.linear.x = 0
	else: 
            cmd_vel.angular.z = 0
        #    cmd_vel.linear.x = 0
	return cmd_vel

    def done(self): 
        cmd_vel = Twist()
        return cmd_vel

    # called periodically to do top-level coordination of behaviors
    def control_callback(self, timer_event=None):

        # initialize vel to 0, 0
	cmd_vel = Twist()
        time_since_stop = rospy.get_rostime() - self.time_of_stop

        cur_pose = self.get_current_pose()
        
        rospy.loginfo(cur_pose)
         
        if cur_pose is None:
            rospy.loginfo("cur_pose is None")
            return
        elif self.start_pose is None:
            self.start_pose = cur_pose    
        
	if self.cones_updated and cur_pose is not None:
            goal_location = self.find_goal_location(cur_pose)
            rospy.loginfo(goal_location)
            # now update self.goal_location
            if goal_location is not None:
                self.goal_in_world_frame = goal_location
                rospy.loginfo('updated current goal!')

        if self.goal_in_world_frame is None:
            rospy.loginfo('Waiting for goal...')
            # For task 2, you might want to uncomment this
            cmd_vel.angular.z = 1.0 # Rotate until we see goal           
        else:
            T_world_from_robot = cur_pose
            T_world_from_goal = self.goal_in_world_frame
            T_world_from_ball = self.ball_in_world_frame

            dist_to_goal = distance_2d(T_world_from_robot.translation(),
                                           T_world_from_goal.translation())

            rospy.loginfo('goal has cones {} and is at {} in world coords, {} meters away'.format(
                    self.goal_in_world_frame,
                    T_world_from_goal.translation(),
                    dist_to_goal,self))
            if self.ball_in_world_frame is None: 
                rospy.loginfo('Waiting for ball...')
            	cmd_vel.angular.z = 1.0
	    else: 
                rospy.loginfo('Ball detected at: {}'.format(self.ball_in_world_frame))
                # TODO: task 2 - set cmd_vel to either turn towards a point
                #                or move towards the point
                if self.goal_in_world_frame is not None and self.ball_in_world_frame is not None:
           	    if self.target_position is None:
			h = 0.5
			midpoint_x = (self.goal_in_world_frame.x + self.ball_in_world_frame.x)/2
                        midpoint_y = (self.goal_in_world_frame.y + self.ball_in_world_frame.y)/2
			magy = (self.ball_in_world_frame.y - self.goal_in_world_frame.y)
			magx = (self.goal_in_world_frame.x - self.ball_in_world_frame.x)
			dist = (magy**2 + magx**2)**.5
			wallpoint_x = midpoint_x + h/dist*magy
			wallpoint_y = midpoint_y + h/dist*magx

			dx = self.ball_in_world_frame.x - wallpoint_x
                        dy = self.ball_in_world_frame.y - wallpoint_y
                        #dtheta = self.ball_in_world_frame.theta - self.goal_in_world_frame.theta
                        wall_to_ball_vec = [dx, dy]
                        c = (dx*dx + dy*dy)**0.5
                        c_prime = c + 1
                        factor = c_prime/c
                        dx_prime = dx*factor
                        dy_prime = dy*factor
                        target_x = wallpoint_x + dx_prime
                        target_y = wallpoint_y + dy_prime
                        self.target_position = [target_x, target_y]

			'''
                        c = ((wallpoint_x - self.ball_in_world_frame.x)**2+(wallpoint_y - self.ball_in_world_frame.y)**2)**.5
                        target_x = self.ball_in_world_frame.x + h/c*0.8
			ydist = self.ball_in_world_frame.y - midpoint_y 
			target_y = self.ball_in_world_frame.y + ydist/c*0.8 
			self.target_position = [target_x, target_y]
			#rospy.loginfo('dx = {} and dy = {}'.format(dx,dy))
			#rospy.loginfo('c = {} and c_prime = {}'.format(c, c_prime))
                        '''
                        #rospy.loginfo('target_position before offset' + str(goal_to_ball_vec))
			rospy.loginfo('wallpoint = ' + str(wallpoint_x) + ', ' + str(wallpoint_y))
			#self.target_position = offset_vec(goal_to_ball_vec, 1)
                        rospy.loginfo('target_position is at' + str(self.target_position))
                        #self.target_position = Transform2D(goal_to_ball_offset[0], goal_to_ball_offset[1], goal_to_ball_offset[2]) 
                    # if * does not work, try .compose_with(other)
		   
                    T_robot_from_world = self.get_current_pose().inverse()
                    #rospy.loginfo('T_robot_from_world is' + str(T_robot_from_world))
                    self.target_in_robot = T_robot_from_world*self.target_position
                    #rospy.loginfo('T_robot_from_world' + str(T_robot_from_world))
                    #rospy.loginfo('self.start_pose' + str(self.start_pose))
                    #rospy.loginfo('self.target_position' + str(self.target_position))    
		    #rospy.loginfo('target_in_robot' + str(self.target_in_robot))
                    #rospy.loginfo('self.start_pose' + str(T_robot_from_world*self.start_pose))
                    #rospy.loginfo('goal in robot' + str(T_robot_from_world*self.start_pose*self.goal_in_world_frame))
                    #rospy.loginfo('ball in robot' + str(T_robot_from_world*self.start_pose*self.ball_in_world_frame))
	            #rospy.loginfo('target in robot' + str(self.target_in_robot))
                    if self.target_reached == False:
                        if self.toDrive == False: 
                            vel = self.turn_to_target(self.target_in_robot)
                            
                        else:
			     
                            vel = self.drive_to_target(self.target_in_robot)
                            
		    else:
                        ball_vec = [self.ball_in_world_frame.x, self.ball_in_world_frame.y]
                        if self.to_kick == False:
			    vel = self.turn_to_ball(T_robot_from_world*ball_vec)
			elif self.start_kick == False:
                            self.time_start_kick = rospy.get_rostime()
                            self.start_kick = True
                            rospy.loginfo(self.time_start_kick.to_sec())
                            vel = None
                        else: 
                            since_kick = rospy.get_rostime() - self.time_start_kick
	           	    rospy.loginfo('time since kick' + str(since_kick.to_sec()))
                            
		    	    if since_kick.to_sec() > 1.3:
				vel = self.done()
			    else: 
		                
                            	vel = self.hit_ball(ball_vec)
                            	rospy.loginfo('Hitting')
	                #    else: 
			
			#vel = self.drive_to_target(ball_vec)
			#cmd_vel = vel
		    if vel is not None:
		    	cmd_vel = vel
                  
            # now filter large changes in velocity before commanding
            # robot - note we don't filter when stopped
        
        cmd_vel.linear.x = filter_vel(self.prev_cmd_vel.linear.x,
                                          cmd_vel.linear.x,
                                          self.MAX_LINEAR_CHANGE)
        cmd_vel.angular.z = filter_vel(self.prev_cmd_vel.angular.z,
                                           cmd_vel.angular.z,
                                           self.MAX_ANGULAR_CHANGE)

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
    

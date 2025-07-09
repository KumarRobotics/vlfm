# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, List, Tuple

import numpy as np

from .base_robot import BaseRobot
from .frame_ids import SpotFrameIds

import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import Twist

MAX_CMD_DURATION = 5

class ROSInterface(object):
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('synchronized_subscriber', anonymous=True)

        # Parameters (can also be loaded from the parameter server)
        self.color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/depth/image_raw")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")

        self.queue_size = rospy.get_param("~queue_size", 10)
        self.slop = rospy.get_param("~slop", 0.1)  # ApproximateTimeSynchronizer slop

        # Create message_filters subscribers
        self.color_sub = Subscriber(self.color_topic, Image)
        self.depth_sub = Subscriber(self.depth_topic, Image)
        self.odom_sub  = Subscriber(self.odom_topic, Odometry)

        # Synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.odom_sub],
            queue_size=self.queue_size,
            slop=self.slop
        )
        self.sync.registerCallback(self.synced_callback)
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        self.odom = None
        self.color = None
        self.depth = None

        rospy.loginfo("SynchronizedSubscriber initialized and listening to topics:")
        rospy.loginfo("  Color: %s", self.color_topic)
        rospy.loginfo("  Depth: %s", self.depth_topic)
        rospy.loginfo("  Odom:  %s", self.odom_topic)

    def synced_callback(self, color_msg, depth_msg, odom_msg):
        """
        Callback with synchronized color image, depth image, and odometry.
        """
        rospy.loginfo("Synchronized callback triggered")
        # Process the messages here
        # Example: just log timestamps
        rospy.loginfo("Color Image timestamp: %s", str(color_msg.header.stamp))
        rospy.loginfo("Depth Image timestamp: %s", str(depth_msg.header.stamp))
        rospy.loginfo("Odometry timestamp: %s", str(odom_msg.header.stamp))
        self.odom = odom_msg
        self.color = color_msg
        self.depth = depth_msg

    def spin(self):
        rospy.loginfo("Spinning...")
        rospy.spin()

    def pub_cmd_vel(vel: float, ang_rate: float):
        cmd_vel = Twist()
        cmd_vel.linear.x = vel
        cmd_vel.angular.z = ang_rate
        self.cmd_vel_pub.publish(cmd_vel)

class JackalRobot(BaseRobot):
    def __init__(self):
        self.ros_interface = ROSInterface()

    @property
    def xy_yaw(self) -> Tuple[np.ndarray, float]:
        """Returns [x, y], yaw"""
        odom = self.ros_interface.odom
        if odom is None:
            return np.array([0., 0.]), 0.
        else:
            x, y, yaw = odom.pose.pose.position.x,
                        odom.pose.pose.position.y,
                        np.atan2(odom.pose.pose.orientation.w, 
                                 odom.pose.pose.orientation.z)
        return np.array([x, y]), yaw

    @property
    def arm_joints(self) -> np.ndarray:
        """Returns current angle for each of the 6 arm joints in radians"""
        raise NotImplementedError

    def get_camera_images(self, camera_source: List[str]) -> Dict[str, np.ndarray]:
        """Returns a dict of images mapping camera ids to images

        Args:
            camera_source (List[str]): List of camera ids to get images from

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping camera ids to images
        """
        imgs = {'color': self.ros_interface.color}
        return imgs

    def command_base_velocity(self, ang_vel: float, lin_vel: float) -> None:
        """Commands the base to execute given angular/linear velocities, non-blocking

        Args:
            ang_vel (float): Angular velocity in radians per second
            lin_vel (float): Linear velocity in meters per second
        """
        # Just make the robot stop moving if both velocities are very low
        self.ros_interface(lin_vel, ang_vel)

    def get_transform(self, frame: str = SpotFrameIds.BODY) -> np.ndarray:
        """Returns the transformation matrix of the robot's base (body) or a link

        Args:
            frame (str, optional): Frame to get the transform of. Defaults to
                SpotFrameIds.BODY.

        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        raise NotImplementedError

    def set_arm_joints(self, joints: np.ndarray, travel_time: float) -> None:
        """Moves each of the 6 arm joints to the specified angle

        Args:
            joints (np.ndarray): Array of 6 angles in radians
            travel_time (float): Time in seconds to reach the specified angles
        """
        raise NotImplementedError

    def open_gripper(self) -> None:
        """Opens the gripper"""
        raise NotImplementedError

    def get_camera_data(self, srcs: List[str]) -> Dict[str, Dict[str, Any]]:
        """Returns a dict that maps each camera id to its image, focal lengths, and
        transform matrix (from camera to global frame).

        Args:
            srcs (List[str]): List of camera ids to get images from

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping camera ids to images
        """
        raise NotImplementedError

    def _camera_response_to_data(self, response: Any) -> Dict[str, Any]:
        image: np.ndarray = image_response_to_cv2(response, reorient=False)
        fx: float = response.source.pinhole.intrinsics.focal_length.x
        fy: float = response.source.pinhole.intrinsics.focal_length.y
        tf_snapshot = response.shot.transforms_snapshot
        camera_frame: str = response.shot.frame_name_image_sensor
        return {
            "image": image,
            "fx": fx,
            "fy": fy,
            "tf_camera_to_global": self.spot.get_transform(from_frame=camera_frame, tf_snapshot=tf_snapshot),
        }

#!/usr/bin/env python3
 
from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray
from controller_msg.msg import PIDTune
from error_msg.msg import Error
import rclpy
from rclpy.node import Node
import numpy as np
import control  # python-control
import sys
from nav_msgs.msg import Odometry
from transformations import euler_from_quaternion
import math
import time
 
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
 
from waypoint_navigation.action import NavTowaypoint
from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray
 
import threading

# --- RC Command Limits ---
MIN_ROLL = 1000
BASE_ROLL = 1500
MAX_ROLL = 2000

MIN_PITCH = 1000
BASE_PITCH = 1500
MAX_PITCH = 2000

MIN_THROTTLE = 1250
BASE_THROTTLE = 1532  # Feed-forward for gravity
MAX_THROTTLE = 2000

class Swift_Pico(Node):
    def __init__(self):
        super().__init__('pico_controller')

        # --- Threading Setup ---
        self.state_lock = threading.Lock()
        self.controller_group = ReentrantCallbackGroup()
        self.action_callback_group = ReentrantCallbackGroup()
        self.whycon_callback_group = ReentrantCallbackGroup()

        # --- Shared State Variables (Protected by Lock) ---
        self.current_pos = np.array([0.0, 0.0, 0.0]) # [x, y, z]
        self.current_vel = np.array([0.0, 0.0, 0.0]) # [vx, vy, vz]
        
        self.desired_pos = np.array([-7.0, 1.0, 20.0]) # Initial desired [x, y, z]
        self.desired_vel = np.array([0.0, 0.0, 0.0])   # Desired [vx, vy, vz]
        
        self.prev_pos_3d = np.array([0.0, 0.0, 0.0])
        self.last_whycon_time = self.get_clock().now()
        self.whycon_received = False
        self.temp_z = 0.0 # For raw error publishing
        # --- End Shared State ---

         # Drone command msg
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = BASE_ROLL
        self.cmd.rc_pitch = BASE_PITCH
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = BASE_THROTTLE
        self.cmd.rc_aux4 = 1000 # Start disarmed

        # --- LQR Controller Setup (6-State) ---
        self.m = 0.152  # Quadcopter mass (kg)
        self.sample_time = 0.065 # Using this from your previous script
        
        # State-space model (Double Integrator for x, y, z)
        # States: [x, y, z, vx, vy, vz] (6 states)
        # Inputs: [Fx, Fy, Fz] (3 inputs)
        
        self.A = np.array([
            [0, 0, 0, 1, 0, 0],  # dx/dt = vx
            [0, 0, 0, 0, 1, 0],  # dy/dt = vy
            [0, 0, 0, 0, 0, 1],  # dz/dt = vz
            [0, 0, 0, 0, 0, 0],  # dvx/dt = 0 (input Fx)
            [0, 0, 0, 0, 0, 0],  # dvy/dt = 0 (input Fy)
            [0, 0, 0, 0, 0, 0]   # dvz/dt = 0 (input Fz)
        ], dtype=float)

        self.B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1/self.m, 0, 0],  # dvx/dt = Fx/m
            [0, 1/self.m, 0],  # dvy/dt = Fy/m
            [0, 0, 1/self.m]   # dvz/dt = Fz/m
        ], dtype=float)

        # Q (State Cost) Matrix (6x6) - Penalizes error
        # Penalties for [x, y, z, vx, vy, vz]
        self.Q = np.diag([
            100.0,  # x pos
            100.0,  # y pos
            120.0,  # z pos (Slightly more important)
            50.0,   # x vel
            50.0,   # y vel
            70.0    # z vel
        ])

        # R (Input Cost) Matrix (3x3) - Penalizes motor effort
        # Penalties for [Fx, Fy, Fz]
        self.R = np.diag([
            20.0,  # Fx (Pitch)
            20.0,  # Fy (Roll)
            10.0   # Fz (Throttle)
        ])

        self.K_lqr, _, __ = ct.lqr(self.A, self.B, self.Q, self.R)
        self.get_logger().info(f"6-State LQR Gain K: {self.K_lqr}")
        
        # Gains to convert LQR Force output to RC commands
        self.throttle_gain = 1.0  # (Tune this)
        self.pitch_gain = 2.0     # (Tune this)
        self.roll_gain = 2.0      # (Tune this)
        # --- End LQR Setup ---
        
        # Publishers
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)

        # Subscribers
        self.create_subscription(
            PoseArray, '/whycon/poses', self.whycon_callback, 1,
            callback_group=self.whycon_callback_group
        )
        
        self._action_server = ActionServer(
            self,
            NavTowaypoint,
            'send_waypoints',
            self.execute_callback,
            callback_group=self.action_callback_group
        )

        # Run Controller periodically
        self.create_timer(
            self.sample_time, 
            self.controller_loop,
            callback_group=self.controller_group
        )
        self.arm()
        
    # ===================== Drone Control =====================
    def disarm(self):
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1500
        self.get_logger().info("Drone disarmed.")
        self.cmd.rc_throttle = 1550 # From pico_server.py
        self.command_pub.publish(self.cmd)

    def arm(self):
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 1500 # From pico_server.py
        self.get_logger().info("Drone armed.")
        self.command_pub.publish(self.cmd)
 
    def whycon_callback(self, msg):
        raw_x = msg.poses[0].position.x
        raw_y = msg.poses[0].position.y
        raw_z = msg.poses[0].position.z
        
        now = self.get_clock().now()
        
        # --- Safely update state ---
        with self.state_lock:
            if not self.whycon_received:
                self.whycon_received = True
                self.prev_pos_3d = [raw_x, raw_y, raw_z]
                self.last_whycon_time = now
            
            dt = (now - self.last_whycon_time).nanoseconds / 1e9
            self.last_whycon_time = now

            self.current_pos[0] =  raw_x
            self.current_pos[1] =  raw_y 
            self.current_pos[2] =  raw_z
            self.temp_z = raw_z # For publishing raw error

            if dt > 1e-6: # Avoid division by zero
                # Calculate velocity for all 3 axes
                self.current_vel[0] = (self.current_pos[0] - self.prev_pos_3d[0]) / dt
                self.current_vel[1] = (self.current_pos[1] - self.prev_pos_3d[1]) / dt
                self.current_vel[2] = (self.current_pos[2] - self.prev_pos_3d[2]) / dt
            
            self.prev_pos_3d = np.copy(self.current_pos)
        # --- End state update ---
 
    def controller_loop(self):
        
        # --- Get a thread-safe local copy of the state ---
        with self.state_lock:
            if not self.whycon_received:
                return # Wait for first sensor reading
            
            local_pos = np.copy(self.current_pos)
            local_vel = np.copy(self.current_vel)
            local_desired_pos = np.copy(self.desired_pos)
            local_desired_vel = np.copy(self.desired_vel)
        
        # --- 1. LQR 6-State Controller ---
        
        # Create the 6-element error vector: e = [x_err, y_err, z_err, vx_err, vy_err, vz_err]
        pos_error = local_desired_pos - local_pos
        vel_error = local_desired_vel - local_vel
        e = np.concatenate([pos_error, vel_error])
        
        # --- Calculate Control Output u = [Fx, Fy, Fz] ---
        # Using the (buggy) logic as requested
        u = -np.dot(self.K_lqr, e)
        
        u_x = u[0] # Force in X
        u_y = u[1] # Force in Y
        u_z = u[2] # Force in Z
        
        # --- 2. Map Control Outputs to RC Commands ---
        
        # Throttle (from Fz)
        throttle_cmd = self.throttle_gain * u_z
        self.cmd.rc_throttle = int(np.clip(BASE_THROTTLE + throttle_cmd, MIN_THROTTLE, MAX_THROTTLE))

        # Pitch (from Fx)
        # Note: Sign ( +/- ) depends on drone setup
        pitch_cmd = self.pitch_gain * u_x
        self.cmd.rc_pitch = int(np.clip(BASE_PITCH + pitch_cmd, MIN_PITCH, MAX_PITCH))

        # Roll (from Fy)
        # Note: Sign ( +/- ) depends on drone setup
        roll_cmd = self.roll_gain * u_y
        self.cmd.rc_roll = int(np.clip(BASE_ROLL + roll_cmd, MIN_ROLL, MAX_ROLL))
        
        # Yaw (uncontrolled)
        self.cmd.rc_yaw = 1500
        
        # --- 3. PUBLISHING ---
        self.cmd.rc_aux4 = 2000 # Keep armed
        self.command_pub.publish(self.cmd)
        
        pos_error_msg = Error()
        pos_error_msg.throttle_error = float(self.temp_z - local_desired_pos[2]) # Raw Z error
        pos_error_msg.pitch_error = float(pos_error[0]) # X error
        pos_error_msg.roll_error = float(pos_error[1])  # Y error
        pos_error_msg.yaw_error  = 0.0
        self.pos_error_pub.publish(pos_error_msg)


    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing waypoint goal...')

        # --- 1. Set desired state (Thread-Safe) ---
        with self.state_lock:
            self.desired_pos[0] = goal_handle.request.waypoint.position.x
            self.desired_pos[1] = goal_handle.request.waypoint.position.y
            self.desired_pos[2] = goal_handle.request.waypoint.position.z
            self.desired_vel = np.array([0.0, 0.0, 0.0]) # Always want to hover
        
        self.get_logger().info(
            f'New Waypoint: X={self.desired_pos[0]:.2f}, '
            f'Y={self.desired_pos[1]:.2f}, Z={self.desired_pos[2]:.2f}'
        )

        # --- 2. Initialize timing and tracking ---
        start_time = self.get_clock().now()
        stable_start = None
        hover_duration = 0.0001
        stable_required = 0.3  # target hover duration
        hover_radius = 0.5    # in meters

        feedback_msg = NavTowaypoint.Feedback()

        # --- 3. Control loop until stabilization ---
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled by client.')
                return NavTowaypoint.Result()
            
            # --- Get a safe, local copy of state for feedback ---
            with self.state_lock:
                local_pos = np.copy(self.current_pos)
                local_desired_pos = np.copy(self.desired_pos)

            # --- Publish feedback (current position) ---
            feedback_msg.current_waypoint.header.stamp = self.get_clock().now().to_msg()
            feedback_msg.current_waypoint.pose.position.x = float(local_pos[0])
            feedback_msg.current_waypoint.pose.position.y = float(local_pos[1])
            feedback_msg.current_waypoint.pose.position.z = float(local_pos[2])
            goal_handle.publish_feedback(feedback_msg)

            # --- Distance from desired waypoint ---
            dist = np.linalg.norm(local_desired_pos - local_pos)

            # --- Stability check ---
            if dist < hover_radius:
                if stable_start is None:
                    stable_start = self.get_clock().now()
                else:
                    hover_duration = (self.get_clock().now() - stable_start).nanoseconds / 1e9
            else:
                stable_start = None
                hover_duration = 0.0

            # --- Break if stabilized ---
            if hover_duration >= stable_required:
                break

            time.sleep(self.sample_time)

        # --- 4. Goal succeeded ---
        goal_handle.succeed()
        total_elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9

        result = NavTowaypoint.Result()
        result.hover_time = int(round(hover_duration, 2))

        self.get_logger().info(
            f'Goal reached and hovered for {hover_duration:.2f}s (Total time {total_elapsed:.2f}s)'
        )

        return result


# ===================== Main =====================
def main(args=None):
    rclpy.init(args=args)
    swift_pico = Swift_Pico()
    # Use 3 threads (Action, Controller, Whycon)
    executor = MultiThreadedExecutor(num_threads=3) 
    executor.add_node(swift_pico)

    try:
        executor.spin()  
    except KeyboardInterrupt:
        swift_pico.disarm()
        swift_pico.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        swift_pico.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
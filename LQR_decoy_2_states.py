import numpy as np
import control as ct  # Python Control Systems Library
import rclpy
from rclpy.node import Node
from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray
from error_msg.msg import Error

MIN_ROLL = 1000
BASE_ROLL = 1500
MAX_ROLL = 2000
SUM_ERROR_ROLL_LIMIT = 5000

MIN_PITCH = 1000
BASE_PITCH = 1500
MAX_PITCH = 2000
SUM_ERROR_PITCH_LIMIT = 5000

MIN_THROTTLE = 1250
BASE_THROTTLE = 1532   # nominal "hover" stick (may need calibration)  hover throttle
MAX_THROTTLE = 2000
SUM_ERROR_THROTTLE_LIMIT = 5000


class Swift_Pico(Node):
    def __init__(self):
        super().__init__('pico_controller')  # initializing ros node with name pico_controller

        self.m = 0.152  # Quadcopter mass (kg)
        self.g = 9.81  # Gravity (m/s^2)

        # sample time
        self.sample_time = 0.0166

        # Variables for velocity calculation
        self.prev_position = np.array([0.0, 0.0, 0.0])
        self.last_time = self.get_clock().now()

        # Flag: only run controller after first whycon reading
        self.whycon_received = False

        # Desired state [z, z_dot]
        self.desired_state = np.array([29 , 0.0], dtype=float)

        # Current state [z ,z_dot]
        self.current_state = np.array([0.0, 0.0], dtype=float)

        # previous state
        self.prev_state = np.array([0.0, 0.0], dtype=float)

        #error matrix
        self.error = np.array([0.0, 0.0], dtype=float)

        self.A =  np.array([0.0 ,1.0 ,0.0, 0.0], dtype=float).reshape(2,2)  # Define the A matrix
        self.B = np.array([0.0, 1/self.m]).reshape(2,1) # Define the B matrix
        self.Q =  np.diag([70, 140])         # size is decided by no. of state variable
        self.R =  np.array([5.0]).reshape(1,1)   # size is defines by no. input variables

        self.K, _, __ = ct.lqr(self.A, self.B, self.Q, self.R)      # Calculate the LQR gain matrix K

        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = BASE_ROLL
        self.cmd.rc_pitch = BASE_PITCH
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = BASE_THROTTLE
        self.cmd.rc_aux4 = 1000

        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)
        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)

     
        self.create_timer(self.sample_time, self.controller)

        self.arm()  # ARMING THE DRONE

    def disarm(self):
        self.cmd.rc_roll = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.get_logger().info("Disarm command sent.")
        self.command_pub.publish(self.cmd)

    def arm(self):
        #self.disarm()
        self.cmd.rc_aux4 = 2000
        self.get_logger().info("Arm command sent.")
        self.command_pub.publish(self.cmd)  # Publishing /drone_command

    def whycon_callback(self, msg):
      self.whycon_received = True
      now = self.get_clock().now()
      dt = (now - self.last_time).nanoseconds / 1e9

      # --- ADD THIS CHECK ---
      if dt < 1e-6:  # if dt is too small, skip this measurement
        return
      # --- END OF CHECK ---

      self.current_state[0] = msg.poses[0].position.z 
      self.z_dot = (self.current_state[0] - self.prev_state[0]) / dt  # calculation for z dot
      self.current_state[1] = self.z_dot  # updating the velocity

      self.prev_state = self.current_state.copy()
      self.last_time = now

    def controller(self):
        self.error[0] = self.desired_state[0] - self.current_state[0] # for position 
        self.error[1] = self.desired_state[1] - self.current_state[1]  # for velocity 
        
        u = -np.dot(self.K, self.error)  # u will be one by one matrix


        self.cmd.rc_throttle = int(np.clip(BASE_THROTTLE + u.item(), MIN_THROTTLE, MAX_THROTTLE)) # main command
        
        #PUBLISHING 
        self.cmd.rc_aux4 = 2000
        pos_error = Error()
        pos_error.throttle_error = self.error[0]
        self.pos_error_pub.publish(pos_error)
        self.command_pub.publish(self.cmd)

        # printing the required 
        self.get_logger().info(f"Position: {self.current_state[0]}")
        self.get_logger().info(f"Position Error: : {self.error[0]}")
        self.get_logger().info(f"Throttle: {self.cmd.rc_throttle}")
        self.get_logger().info(f"u: : {u}")

def main(args=None):
	rclpy.init(args=args)
	swift_pico = Swift_Pico()
 
	try:
		rclpy.spin(swift_pico)
	except KeyboardInterrupt:
		swift_pico.get_logger().info('KeyboardInterrupt, shutting down.\n')
	finally:
		swift_pico.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()
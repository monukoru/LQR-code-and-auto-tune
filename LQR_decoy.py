#!/usr/bin/env python3

from rc_msgs.msg import RCMessage
import numpy as np
from rc_msgs.srv import CommandBool
from geometry_msgs.msg import PoseArray
from controller_msg.msg import PIDTune
from error_msg.msg import Error
from crsf_msgs.msg import BatterySensor
import control as ct  # Python Control Systems Library
import time
#from sensor_msgs.msg import BatteryState

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import rclpy
from rclpy.node import Node
from scipy.signal import butter, lfilter_zi, lfilter # [ADDED] Import for Butterworth

class Swift_Pico(Node):
    def __init__(self):
        super().__init__('pico_controller')
        self.pid_callback_group = ReentrantCallbackGroup()     

        # -------------------- Drone Params --------------------
        self.m = 0.152  # Quadcopter mass (kg)
        self.g = 9.81  # Gravity (m/s^2)
        
        # ----------------------- whycon bool ------------------

        self.whycon_received = True

        # -------------------- States --------------------

        self.current_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # [x, y, z, x_dot, y_dot, z_dot]  all the co-ordinates come as cm from whycon

        self.desired_state = [-0.55, 0.0, 1.9, 0.0, 0.0, 0.0] # [x, y, z, x_dot, y_dot, z_dot]

        # ------------------- matrix -----------------------
        self.error = np.zeros((6,1), dtype=np.float16)

        self.A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float16)

        self.B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [self.g, 0, 0],
            [0, -self.g, 0],
            [0, 0, 1/self.m]
        ], dtype=np.float16)

        self.Q = np.diag([5, 5, 10, 1, 1, 2])

        self.R = np.diag([1, 1, 2])

        self.K, _, _ = ct.lqr(self.A, self.B, self.Q, self.R)

        # -------------------- Butterworth Filter Params [ADDED] --------------------

        # Parameters taken from your reference file (adjust as needed)
        self.fs = 45.0       # sampling frequency in Hz
        self.cutoff = 3.5    # cutoff frequency in Hz
        self.order = 2       # filter order

        # Initialize filter coefficients

        self.b, self.a = butter(self.order, self.cutoff / (0.5 * self.fs), btype='low')

        self.zi_x = lfilter_zi(self.b, self.a) * 0.0
        self.zi_y = lfilter_zi(self.b, self.a) * 0.0
        self.zi_z = lfilter_zi(self.b, self.a) * 0.0

        # -------------------- RC Command --------------------

        self.cmd = RCMessage()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500

        # Limits for RC commands
        self.max_values = [1530, 1530, 1530]
        self.min_values = [1440, 1440, 1465]

        # -------------------- Timing --------------------

        self.sample_time = 0.0166

        # -------------------- Output Smoothing (EMA) --------------------

        self.output_filtered = np.zeros((1,6), dtype=np.float16)
        self.ema_alpha = 0.60

        # -------------------- Battery Compensation --------------------

        self.battery_voltage = 4.2 # Initialize as full
        self.throttle_base = 1500  # Default Hover PWM

        #--------------------- previous state --------------------------

        self.prev_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float) #[x, y, z, x_dot, y_dot, z_dot]
        self.last_time = self.get_clock().now()

        # -------------------- Publishers --------------------

        self.command_pub = self.create_publisher(RCMessage, '/drone/rc_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)

        # -------------------- Service Client --------------------

        self.arm_client = self.create_client(CommandBool, '/drone/cmd/arming')

        # -------------------- Subscribers --------------------

        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)
        #self.create_subscription(BatterySensor, '/drone/battery_info', self.battery_callback_cascade, 10)
        self.arm_drone()
        # Start PID Loop
        self.create_timer(self.sample_time, self.controller)

     
# -------------------- ARM / DISARM --------------------

    def arm_drone(self):
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for arming service...')
        req = CommandBool.Request()
        req.value = True
        self.arm_client.call_async(req)
        self.get_logger().info('ARMING REQUEST SENT')

    def disarm_drone(self):
        req = CommandBool.Request()
        req.value = False
        self.arm_client.call_async(req)
        self.cmd.rc_throttle = 1000
        self.command_pub.publish(self.cmd)
        self.get_logger().info('DISARMED')


    #---------------- whycon function --------------

    def whycon_callback(self, msg):
      self.whycon_received = True
      now = self.get_clock().now()
      dt = (now - self.last_time).nanoseconds / 1e9

      # --- ADD THIS CHECK ---
      if dt < 1e-6:  # if dt is too small, skip this measurement
        return
      
      x = msg.poses[0].position.x *0.01
      y = msg.poses[0].position.y *0.01
      z = msg.poses[0].position.z *0.01

      self.current_state[0] = x
      self.current_state[1] = y
      self.current_state[2] = z

      x_filt, self.zi_x = lfilter(self.b, self.a, [x], zi=self.zi_x)
      y_filt, self.zi_y = lfilter(self.b, self.a, [y], zi=self.zi_y)
      z_filt, self.zi_z = lfilter(self.b, self.a, [z], zi=self.zi_z)
 
     # Update state vector
      self.current_state[0] = x_filt[0]
      self.current_state[1] = y_filt[0]
      self.current_state[2] = z_filt[0]

      self.x_dot = (self.current_state[0] - self.prev_state[0]) / dt  # calculation for x dot
      self.y_dot = (self.current_state[1] - self.prev_state[1]) / dt  # calculation for y dot      
      self.z_dot = (self.current_state[2] - self.prev_state[2]) / dt  # calculation for z dot

      self.current_state[3] = self.x_dot
      self.current_state[4] = self.y_dot
      self.current_state[5] = self.z_dot

      self.prev_state = self.current_state.copy()
      self.last_time = now

    def controller(self):
        for i in range(6):  # erro calculation of position and error
            self.error[i,0] = self.desired_state[i] - self.current_state[i]
        
        u = -np.dot(self.K, self.error) # This matrix will be of size 1X3 [roll, pitch, throttle]

        self.cmd.rc_roll = int(np.clip(1500 + u[0, 0],self.min_values[0] ,self.max_values[0] ))
        self.cmd.rc_pitch = int(np.clip(1500 + u[1, 0],self.min_values[1] ,self.max_values[1] ))         
        self.cmd.rc_throttle = int(np.clip(1500 + u[2, 0],self.min_values[2] ,self.max_values[2] ))

        #PUBLISHING 
        self.cmd.rc_aux4 = 2000
        pos_error = Error()
        pos_error.x_error = float(self.error[0])
        pos_error.y_error = float(self.error[1])
        pos_error.z_error = float(self.error[2])
        self.pos_error_pub.publish(pos_error)
        self.command_pub.publish(self.cmd)
        

#-------------------MAIN / FUNCTION -----------------   

def main(args=None):
    rclpy.init(args=args)
    swift_pico = Swift_Pico()
    executor = MultiThreadedExecutor()
    executor.add_node(swift_pico)

    try:
        executor.spin()
    except KeyboardInterrupt:
        swift_pico.get_logger().info('KeyboardInterrupt, shutting down.')
        swift_pico.disarm_drone()
    finally:
        swift_pico.destroy_node()
        rclpy.shutdown()
        swift_pico.disarm_drone()

if __name__ == '__main__':
    main()

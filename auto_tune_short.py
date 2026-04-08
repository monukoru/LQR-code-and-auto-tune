#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time
import signal
import sys

from geometry_msgs.msg import PoseArray
from rc_msgs.msg import RCMessage
from rc_msgs.srv import CommandBool
from error_msg.msg import Error


class KrishiAutoTune(Node):
    def __init__(self):
        super().__init__('krishi_auto_tune')

        #  STATE 
        self.current = [0.0, 0.0, 0.0]
        self.desired = [0.0, 0.0, 19.0]   # hover altitude fixed

        #  PID 
        self.Kp = [0.0, 0.0, 0.0]
        self.Ki = [0.0, 0.0, 0.0]
        self.Kd = [0.0, 0.0, 0.0]

        self.integral = [0.0, 0.0, 0.0]
        self.prev_error = [0.0, 0.0, 0.0]
        self.d_filter_state = [0.0, 0.0, 0.0]

        #  RC 
        self.cmd = RCMessage()
        self.neutral = 1500
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500

        #  AUTO TUNE 
        self.axis = 0              # 0=roll, 1=pitch
        self.kp = 0.0
        self.kp_step = 0.02
        self.kp_max = 0.6

        self.step_amp = 0.2        # meters
        self.test_time = 1.5
        self.start_time = None
        self.error_log = []

        #  TIMING 
        self.dt = 0.033
        self.last_time = time.time()

        #  ROS 
        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_cb, 1)
        self.cmd_pub = self.create_publisher(RCMessage, '/drone/rc_command', 10)
        self.err_pub = self.create_publisher(Error, '/pos_error', 10)
        self.arm_client = self.create_client(CommandBool, '/drone/cmd/arming')

        self.create_timer(self.dt, self.control_loop)

        self.arm()
        signal.signal(signal.SIGINT, self.exit_handler)

        self.get_logger().info("AUTO TUNING NODE STARTED")

    # CALLBACKS

    def whycon_cb(self, msg):
        if msg.poses:
            self.current[0] = msg.poses[0].position.x
            self.current[1] = msg.poses[0].position.y
            self.current[2] = msg.poses[0].position.z

    # PID + AUTO TUNE LOOP


    def control_loop(self):
        now = time.time()
        dt = now - self.last_time
        if dt < self.dt:
            return
        self.last_time = now

        error = [
            self.desired[0] - self.current[0],
            self.desired[1] - self.current[1],
            self.desired[2] - self.current[2],
        ]

        #  AUTO TUNE 
        self.auto_tune(error)

        #  PID 
        out = [0.0, 0.0, 0.0]
        for i in range(3):
            self.integral[i] += error[i] * dt
            self.integral[i] = max(-200, min(200, self.integral[i]))
            d = (error[i] - self.prev_error[i]) / dt
            self.prev_error[i] = error[i]

            out[i] = (
                self.Kp[i] * error[i] +
                self.Ki[i] * self.integral[i] +
                self.Kd[i] * d
            )

        self.cmd.rc_roll = int(1500 + out[0])
        self.cmd.rc_pitch = int(1500 + out[1])
        self.cmd.rc_throttle = int(1500 + out[2])

        self.cmd_pub.publish(self.cmd)

    # AUTO TUNE CORE

    def auto_tune(self, error):
        if self.start_time is None:
            self.kp += self.kp_step

            if self.kp > self.kp_max:
                self.finish()
                return

            self.Kp[self.axis] = self.kp
            self.Ki[self.axis] = 0.0
            self.Kd[self.axis] = 0.0

            self.integral[self.axis] = 0.0
            self.prev_error[self.axis] = 0.0

            self.desired[self.axis] += self.step_amp
            self.start_time = time.time()
            self.error_log = []

            self.get_logger().info(
                f"TUNING AXIS {self.axis} | Kp={self.kp:.3f}"
            )

        self.error_log.append(error[self.axis])

        if time.time() - self.start_time > self.test_time:
            self.desired[self.axis] -= self.step_amp

            if self.unstable(self.error_log):
                self.Kp[self.axis] -= self.kp_step

                self.get_logger().warn(
                    f"OSCILLATION → FINAL Kp[{self.axis}] = {self.Kp[self.axis]:.3f}"
                )

                if self.axis == 0:
                    self.axis = 1
                    self.kp = 0.0
                    self.start_time = None
                    self.get_logger().info("STARTING PITCH TUNE")
                    return
                else:
                    self.finish()
                    return

            self.start_time = None

    def unstable(self, data):
        zero_cross = 0
        for i in range(1, len(data)):
            if data[i-1] * data[i] < 0:
                zero_cross += 1
        return zero_cross > 3

    # ARM / EXIT

    def arm(self):
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for arm service")
        req = CommandBool.Request()
        req.value = True
        self.arm_client.call_async(req)

    def finish(self):
        self.get_logger().info("AUTO TUNING COMPLETE")
        self.get_logger().info(f"FINAL GAINS: ROLL Kp={self.Kp[0]:.3f}, PITCH Kp={self.Kp[1]:.3f}")
        self.exit_handler(None, None)

    def exit_handler(self, sig, frame):
        self.cmd.rc_throttle = 1000
        self.cmd_pub.publish(self.cmd)
        rclpy.shutdown()
        sys.exit(0)


def main(args=None):
    rclpy.init(args=args)
    node = KrishiAutoTune()
    rclpy.spin(node)


if __name__ == '__main__':
    main()

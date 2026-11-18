#!/usr/bin/env python3
"""Log nav_msgs/Odometry (FAST-LIO /Odometry) to a text file: t x y z qx qy qz qw (TUM format)."""
import sys, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry

class L(Node):
    def __init__(self, path):
        super().__init__("odom_logger")
        self.f = open(path, "w"); self.n = 0
        self.create_subscription(Odometry, "/Odometry", self.cb,
                                 QoSProfile(depth=10000, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST))
    def cb(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.f.write(f"{t:.9f} {p.x:.6f} {p.y:.6f} {p.z:.6f} {q.x:.8f} {q.y:.8f} {q.z:.8f} {q.w:.8f}\n"); self.n += 1
        if self.n % 200 == 0: self.f.flush()

rclpy.init(); n = L(sys.argv[1])
try: rclpy.spin(n)
except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException): pass
finally: n.f.close()

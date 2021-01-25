import rospy
from lowpass import LowPassFilter
from pid import PID
from yaw_controller import YawController
import numpy as np


GAS_DENSITY = 2.858
STOP_VELOCITY = 0.1


class Controller(object):
    def __init__(self, vehicle_params):
        self.vehicle_params = vehicle_params
        self.lpf_s = LowPassFilter(tau=3, ts=1)
        self.lpf_t = LowPassFilter(tau=3, ts=1)
        self.pid = PID(kp=1.5, ki=0.001, kd=0., mn=vehicle_params.decel_limit, mx=vehicle_params.accel_limit)
        self.yaw_controller = YawController(
            wheel_base=vehicle_params.wheel_base,
            steer_ratio=vehicle_params.steer_ratio,
            min_speed=vehicle_params.min_speed,
            max_lat_accel=vehicle_params.max_lat_accel,
            max_steer_angle=vehicle_params.max_steer_angle
        )

        self.decel_limit = vehicle_params.decel_limit
        self.wheel_radius = vehicle_params.wheel_radius
        self.total_mass = vehicle_params.vehicle_mass + (vehicle_params.fuel_capacity * GAS_DENSITY)

    def torque(self, accel):
        return accel * self.total_mass * self.wheel_radius

    def control(self, delta_time, current_velocity, twist_cmd):
        v_error = twist_cmd.twist.linear.x - current_velocity.twist.linear.x
        accel = self.pid.step(v_error, delta_time)

        steer = self.yaw_controller.get_steering(twist_cmd.twist.linear.x, twist_cmd.twist.angular.z, current_velocity.twist.linear.x)

        if current_velocity.twist.linear.x < STOP_VELOCITY and np.isclose(twist_cmd.twist.linear.x, 0.):
            return 0., self.torque(self.decel_limit), steer
        
        else:
        
            if accel > 0:
                return accel, 0., steer
        
            else:
                torque = self.torque(-accel)
                return 0., torque, steer

    def reset(self):
        self.pid.reset()
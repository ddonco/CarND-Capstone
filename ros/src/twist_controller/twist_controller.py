import rospy
from lowpass import LowPassFilter
from pid import PID
from yaw_controller import YawController


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_params):
        self.vehicle_params = vehicle_params
        self.lpf_s = LowPassFilter(tau=3, ts=1)
        self.lpf_t = LowPassFilter(tau=3, ts=1)
        self.pid = PID(kp=5, ki=0.5, kd=0.5, mn=vehicle_params.decel_limit, mx=vehicle_params.accel_limit)
        self.yaw_controller = YawController(
            wheel_base=vehicle_params.wheel_base,
            steer_ratio=vehicle_params.steer_ratio,
            min_speed=vehicle_params.min_speed,
            max_lat_accel=vehicle_params.max_lat_accel,
            max_steer_angle=vehicle_params.max_steer_angle
        )

    def control(self, del_time, velocity_current, twist_cmd):
        # Return throttle, brake, steer
        velocity_linear = abs(twist_cmd.twist.linear.z)
        velocity_angular = twist_cmd.twist.angular.z
        velocity_error = velocity_linear - velocity_current.twist.linear.x

        steer_next = self.yaw_controller.get_steering(velocity_linear, velocity_angular, velocity_current.twist.linear.x)
        steer_next = self.lpf_s.filt(steer_next)

        accel = self.pid.step(velocity_error, del_time)
        accel = self.lpf_t.filt(accel)

        if accel > 0.:
            throttle = accel
            brake = 0.
        
        else:
            throttle = 0.
            decel = 0.

            if decel < self.vehicle_params.brake_deadband:
                decel = 0.

            vehicle_net_mass = self.vehicle_params.vehicle_mass + self.vehicle_params.fuel_capacity * GAS_DENSITY
            brake = decel * vehicle_net_mass * self.vehicle_params.wheel_radius

        return throttle, brake, steer_next

    def reset(self):
        self.pid.reset()
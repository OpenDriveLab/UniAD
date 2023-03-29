#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from casadi import DM, Opti, OptiSol, cos, diff, sin, sumsqr, vertcat
Pose = Tuple[float, float, float]  # (x, y, yaw)


class MotionNonlinearSmoother:
    """
    Smoothing a set of xy observations with a vehicle dynamics model.
    Solved with direct multiple-shooting.
    modified from https://github.com/motional/nuplan-devkit
    :param trajectory_len: trajectory length
    :param dt: timestep (sec)
    """

    def __init__(self, trajectory_len: int, dt: float):
        """
        :param trajectory_len: the length of trajectory to be optimized.
        :param dt: the time interval between trajectory points.
        """
        self.dt = dt
        self.trajectory_len = trajectory_len
        self.current_index = 0
        # Use a array of dts to make it compatible to situations with varying dts across different time steps.
        self._dts: npt.NDArray[np.float32] = np.asarray(
            [[dt] * trajectory_len])
        self._init_optimization()

    def _init_optimization(self) -> None:
        """
        Initialize related variables and constraints for optimization.
        """
        self.nx = 4  # state dim
        self.nu = 2  # control dim

        self._optimizer = Opti()  # Optimization problem
        self._create_decision_variables()
        self._create_parameters()
        self._set_dynamic_constraints()
        self._set_state_constraints()
        self._set_control_constraints()
        self._set_objective()

        # Set default solver options (quiet)
        self._optimizer.solver(
            "ipopt", {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"})

    def set_reference_trajectory(self, x_curr: Sequence[float], reference_trajectory: Sequence[Pose]) -> None:
        """
        Set the reference trajectory that the smoother is trying to loosely track.
        :param x_curr: current state of size nx (x, y, yaw, speed)
        :param reference_trajectory: N+1 x 3 reference, where the second dim is for (x, y, yaw)
        """
        self._check_inputs(x_curr, reference_trajectory)

        self._optimizer.set_value(self.x_curr, DM(x_curr))
        self._optimizer.set_value(self.ref_traj, DM(reference_trajectory).T)
        self._set_initial_guess(x_curr, reference_trajectory)

    def set_solver_optimizerons(self, options: Dict[str, Any]) -> None:
        """
        Control solver options including verbosity.
        :param options: Dictionary containing optimization criterias
        """
        self._optimizer.solver("ipopt", options)

    def solve(self) -> OptiSol:
        """
        Solve the optimization problem. Assumes the reference trajectory was already set.
        :return Casadi optimization class
        """
        return self._optimizer.solve()

    def _create_decision_variables(self) -> None:
        """
        Define the decision variables for the trajectory optimization.
        """
        # State trajectory (x, y, yaw, speed)
        self.state = self._optimizer.variable(self.nx, self.trajectory_len + 1)
        self.position_x = self.state[0, :]
        self.position_y = self.state[1, :]
        self.yaw = self.state[2, :]
        self.speed = self.state[3, :]

        # Control trajectory (curvature, accel)
        self.control = self._optimizer.variable(self.nu, self.trajectory_len)
        self.curvature = self.control[0, :]
        self.accel = self.control[1, :]

        # Derived control and state variables, dt[:, 1:] becuases state vector is one step longer than action.
        self.curvature_rate = diff(self.curvature) / self._dts[:, 1:]
        self.jerk = diff(self.accel) / self._dts[:, 1:]
        self.lateral_accel = self.speed[: self.trajectory_len] ** 2 * \
            self.curvature

    def _create_parameters(self) -> None:
        """
        Define the expert trjactory and current position for the trajectory optimizaiton.
        """
        self.ref_traj = self._optimizer.parameter(
            3, self.trajectory_len + 1)  # (x, y, yaw)
        self.x_curr = self._optimizer.parameter(self.nx, 1)

    def _set_dynamic_constraints(self) -> None:
        r"""
        Set the system dynamics constraints as following:
          dx/dt = f(x,u)
          \dot{x} = speed * cos(yaw)
          \dot{y} = speed * sin(yaw)
          \dot{yaw} = speed * curvature
          \dot{speed} = accel
        """
        state = self.state
        control = self.control
        dt = self.dt

        def process(x: Sequence[float], u: Sequence[float]) -> Any:
            """Process for state propagation."""
            return vertcat(x[3] * cos(x[2]), x[3] * sin(x[2]), x[3] * u[0], u[1])

        for k in range(self.trajectory_len):  # loop over control intervals
            # Runge-Kutta 4 integration
            k1 = process(state[:, k], control[:, k])
            k2 = process(state[:, k] + dt / 2 * k1, control[:, k])
            k3 = process(state[:, k] + dt / 2 * k2, control[:, k])
            k4 = process(state[:, k] + dt * k3, control[:, k])
            next_state = state[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self._optimizer.subject_to(
                state[:, k + 1] == next_state)  # close the gaps

    def _set_control_constraints(self) -> None:
        """Set the hard control constraints."""
        curvature_limit = 1.0 / 5.0  # 1/m
        self._optimizer.subject_to(
            self._optimizer.bounded(-curvature_limit, self.curvature, curvature_limit))
        accel_limit = 4.0  # m/s^2
        self._optimizer.subject_to(
            self._optimizer.bounded(-accel_limit, self.accel, accel_limit))

    def _set_state_constraints(self) -> None:
        """Set the hard state constraints."""
        # Constrain the current time -- NOT start of history
        # initial boundary condition
        self._optimizer.subject_to(
            self.state[:, self.current_index] == self.x_curr)

        max_speed = 35.0  # m/s
        self._optimizer.subject_to(self._optimizer.bounded(
            0.0, self.speed, max_speed))  # only forward
        max_yaw_rate = 1.75  # rad/s
        self._optimizer.subject_to(
            self._optimizer.bounded(-max_yaw_rate, diff(self.yaw) / self._dts, max_yaw_rate))
        max_lateral_accel = 4.0  # m/s^2, assumes circular motion acc_lat = speed^2 * curvature
        self._optimizer.subject_to(
            self._optimizer.bounded(
                -max_lateral_accel, self.speed[:, : self.trajectory_len] ** 2 *
                self.curvature, max_lateral_accel
            )
        )

    def _set_objective(self) -> None:
        """Set the objective function. Use care when modifying these weights."""
        # Follow reference, minimize control rates and absolute inputs
        alpha_xy = 1.0
        alpha_yaw = 0.1
        alpha_rate = 0.08
        alpha_abs = 0.08
        alpha_lat_accel = 0.06
        cost_stage = (
            alpha_xy *
            sumsqr(self.ref_traj[:2, :] -
                   vertcat(self.position_x, self.position_y))
            + alpha_yaw * sumsqr(self.ref_traj[2, :] - self.yaw)
            + alpha_rate * (sumsqr(self.curvature_rate) + sumsqr(self.jerk))
            + alpha_abs * (sumsqr(self.curvature) + sumsqr(self.accel))
            + alpha_lat_accel * sumsqr(self.lateral_accel)
        )

        # Take special care with the final state
        alpha_terminal_xy = 1.0
        alpha_terminal_yaw = 40.0  # really care about final heading to help with lane changes
        cost_terminal = alpha_terminal_xy * sumsqr(
            self.ref_traj[:2, -1] -
            vertcat(self.position_x[-1], self.position_y[-1])
        ) + alpha_terminal_yaw * sumsqr(self.ref_traj[2, -1] - self.yaw[-1])

        self._optimizer.minimize(
            cost_stage + self.trajectory_len / 4.0 * cost_terminal)

    def _set_initial_guess(self, x_curr: Sequence[float], reference_trajectory: Sequence[Pose]) -> None:
        """Set a warm-start for the solver based on the reference trajectory."""
        self._check_inputs(x_curr, reference_trajectory)

        # Initialize state guess based on reference
        self._optimizer.set_initial(self.state[:3, :], DM(
            reference_trajectory).T)  # (x, y, yaw)
        self._optimizer.set_initial(self.state[3, :], DM(x_curr[3]))  # speed

    def _check_inputs(self, x_curr: Sequence[float], reference_trajectory: Sequence[Pose]) -> None:
        """Raise ValueError if inputs are not of proper size."""
        if len(x_curr) != self.nx:
            raise ValueError(
                f"x_curr length {len(x_curr)} must be equal to state dim {self.nx}")

        if len(reference_trajectory) != self.trajectory_len + 1:
            raise ValueError(
                f"reference traj length {len(reference_trajectory)} must be equal to {self.trajectory_len + 1}"
            )

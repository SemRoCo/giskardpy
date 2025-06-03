import random
from typing import Optional

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import PrefixName
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.tasks.task import Task, WEIGHT_ABOVE_CA
from giskardpy.symbol_manager import symbol_manager


class WiggleInsert(Task):

    def __init__(self, *,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 hole_point: cas.TransMatrix,
                 noise_translation: float,
                 noise_angle: float,
                 down_velocity: float,
                 hole_normal: Optional[cas.Vector3] = None,
                 threshold: float = 0.01,
                 random_walk: bool = True,
                 vector_momentum_factor: float = 0.9,
                 angular_momentum_factor: float = 0.9,
                 center_pull_strength_angle: float = 0.1,
                 center_pull_strength_vector: float = 0.25,
                 weight: float = WEIGHT_ABOVE_CA,
                 name: Optional[str] = None,
                 plot: bool = True):
        """
        Press down while wiggling the end effector.
        :param root_link:
        :param tip_link:
        :param hole_point: Center point of the hole
        :param hole_normal: Vector perpendicular to the hole. default = z-axis of map
        :param threshold: threshold for distance to hole_point to end task
        :param noise_translation: describes how strong the translation wiggle is.
        :param noise_angle: describes how strong the angular wiggle is.
        :param random_walk: determines if random walk or random sample strategy is used
        :param vector_momentum_factor: (only when random_walk=True) Higher value increases influence of momentum,
                                                                    creating smoother but less random translation movement
        :param angular_momentum_factor: (only when random_walk=True) Higher value increases influence of momentum,
                                                                     creating smoother but less random angular movement
        :param center_pull_strength_angle: (only when random_walk=True) Forces angular movement faster back towards
                                                                        starting angle
        :param center_pull_strength_vector: (only when random_walk=True) Forces translation movement faster back towards
                                                                         hole_point
        """
        if hole_normal is None:
            hole_normal = cas.Vector3().from_xyz(0, 0, -1, god_map.world.root_link_name)
        self.root_link = root_link
        self.tip_link = tip_link
        self.hole_point = hole_point
        self.hole_normal = hole_normal
        self.noise_translation = noise_translation
        self.noise_angle = noise_angle
        super().__init__(name=name, plot=plot)

        # Random-Sample works better with control_dt and Random-Walk with throttling using self.dt in my testing
        if random_walk:
            self.dt = god_map.qp_controller.control_dt
        else:
            self.dt = god_map.qp_controller.control_dt
        self.hz = 1 / self.dt

        if random_walk:
            vector_function = '.get_rand_walk_vector()'
            angle_function = '.get_rand_walk_angle()'
        else:
            vector_function = '.get_rand_vector()'
            angle_function = '.get_rand_angle()'

        # Init-Values for angular random walk
        self.current_angle = 0
        self.center_angle = 0
        self.angular_momentum = 0.0
        self.last_angular_change = 0
        self.angular_momentum_factor = angular_momentum_factor
        self.center_pull_strength_angle = center_pull_strength_angle

        # Init-Values for vector random walk
        self.current_vector = np.zeros(3)
        self.center_vector = np.zeros(3)
        self.vector_momentum = np.zeros(3)
        self.last_vector_change = 0
        self.vector_momentum_factor = vector_momentum_factor
        self.center_pull_strength_vector = center_pull_strength_vector

        v1, v2 = self.calculate_vectors(self.hole_normal.to_np()[:3])
        self.v1 = cas.Vector3().from_xyz(*v1, reference_frame=hole_normal.reference_frame)
        self.v2 = cas.Vector3().from_xyz(*v2, reference_frame=hole_normal.reference_frame)

        r_P_c = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()
        r_P_g = god_map.world.transform(self.root_link, self.hole_point).to_position()

        rand_v = symbol_manager.get_expr(self.ref_str +
                                         vector_function,
                                         input_type_hint=np.ndarray,
                                         output_type_hint=cas.Vector3)

        r_P_g_rand = r_P_g + rand_v

        self.add_point_goal_constraints(frame_P_current=r_P_c,
                                        frame_P_goal=r_P_g_rand,
                                        reference_velocity=down_velocity,
                                        weight=weight,
                                        name=f'{name}_point_goal')

        angle = symbol_manager.get_expr(self.ref_str +
                                        angle_function,
                                        input_type_hint=float,
                                        output_type_hint=cas.Symbol)

        tip_V_hole_normal = god_map.world.transform(self.tip_link, self.hole_normal)
        tip_R_hole_normal = cas.RotationMatrix.from_axis_angle(angle=angle,
                                                               axis=tip_V_hole_normal)
        root_R_hole_normal = god_map.world.compute_fk(self.root_link, self.tip_link).dot(tip_R_hole_normal)

        c_R_r_eval = god_map.world.compose_fk_evaluated_expression(self.tip_link, self.root_link).to_rotation()
        r_T_c = god_map.world.compose_fk_expression(self.root_link, self.tip_link)
        r_R_c = r_T_c.to_rotation()

        self.add_rotation_goal_constraints(frame_R_current=r_R_c,
                                           frame_R_goal=root_R_hole_normal,
                                           current_R_frame_eval=c_R_r_eval,
                                           reference_velocity=down_velocity,
                                           weight=weight + 1)

        god_map.debug_expression_manager.add_debug_expression(name='r_P_g',
                                                              expression=r_P_g)
        god_map.debug_expression_manager.add_debug_expression(name='r_P_g_rand',
                                                              expression=r_P_g_rand)
        god_map.debug_expression_manager.add_debug_expression(name='root_R_hole_normal',
                                                              expression=root_R_hole_normal)
        god_map.debug_expression_manager.add_debug_expression(name='tip_R_hole_normal',
                                                              expression=tip_R_hole_normal)

        dist = cas.euclidean_distance(r_P_c, r_P_g)
        end = cas.less_equal(dist, threshold)

        self.observation_expression = end

    def get_rand_angle(self) -> float:
        now = god_map.time
        if now - self.last_angular_change >= self.dt:
            self.last_angular_change = now

            self.current_angle = ((random.random() - 0.5) * self.noise_angle) / self.hz
        return self.current_angle

    def get_rand_walk_angle(self):
        now = god_map.time
        if now - self.last_angular_change >= self.dt:
            self.last_angular_change = now
            random_angular_change = ((random.random() - 0.5) * self.noise_angle) / self.hz

            # Update angular momentum (weighted average of previous momentum and new random change)
            self.angular_momentum = (self.angular_momentum_factor * self.angular_momentum +
                                     (1 - self.angular_momentum_factor) * random_angular_change)

            self.current_angle += self.angular_momentum
            angle_diff = self.center_angle - self.current_angle
            angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi  # Normalize angle difference to [-pi, pi] range
            self.current_angle += angle_diff * self.center_pull_strength_angle
            self.current_angle = self.current_angle % (2 * np.pi)  # Normalize current angle to [0, 2pi] range

        return self.current_angle

    def get_rand_vector(self) -> cas.Vector3:
        now = god_map.time
        if now - self.last_vector_change >= self.dt:
            self.last_vector_change = now

            self.current_vector = (((random.random() - 0.5) * self.noise_translation * self.v1.to_np()
                                    + (random.random() - 0.5) * self.noise_translation * self.v2.to_np())
                                   / self.hz)
        return self.current_vector

    def get_rand_walk_vector(self) -> cas.Vector3:
        now = god_map.time
        if now - self.last_vector_change >= self.dt:
            self.last_vector_change = now

            random_vector_change = (((random.random() - 0.5) * self.noise_translation * self.v1.to_np()[:3]
                                     + (random.random() - 0.5) * self.noise_translation * self.v2.to_np()[:3])
                                    / self.hz)
            self.vector_momentum = (self.vector_momentum_factor * self.vector_momentum +
                                    (1 - self.vector_momentum_factor) * random_vector_change)
            self.current_vector += self.vector_momentum
            vector_diff = self.center_vector - self.current_vector
            self.current_vector += vector_diff * self.center_pull_strength_vector

        return self.current_vector

    def calculate_vectors(self, n: np.ndarray):
        if abs(n[0]) >= abs(n[1]):
            u = np.array([-n[2], 0, n[0]])
        else:
            u = np.array([0, -n[2], n[1]])

        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)

        return u, v

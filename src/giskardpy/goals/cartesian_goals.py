from __future__ import division

import numpy as np
from geometry_msgs.msg import PointStamped, PoseStamped, QuaternionStamped, Vector3Stamped

from giskardpy import casadi_wrapper as w, identifier
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.goals.pointing import PointingDiffDrive
from giskardpy.god_map import GodMap


class CartesianPosition(Goal):
    def __init__(self, root_link: str, tip_link: str, goal_point: PointStamped, reference_velocity: float = None,
                 max_velocity: float = 0.2, weight: float = WEIGHT_ABOVE_CA, **kwargs):
        """
        This goal will use the kinematic chain between root and tip link to achieve a goal position for tip link.
        :param root_link: root link of kinematic chain
        :param tip_link: tip link of kinematic chain
        :param goal: the goal, orientation part will be ignored
        :param max_velocity: m/s
        :param reference_velocity: m/s
        :param weight: default WEIGHT_ABOVE_CA
        """
        super(CartesianPosition, self).__init__(**kwargs)
        if reference_velocity is None:
            reference_velocity = max_velocity
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_point = self.transform_msg(self.root_link, goal_point)
        self.reference_velocity = reference_velocity
        self.max_velocity = max_velocity
        self.weight = weight
        if self.max_velocity is not None:
            self.add_constraints_of_goal(TranslationVelocityLimit(root_link=root_link,
                                                                  tip_link=tip_link,
                                                                  weight=weight,
                                                                  max_velocity=max_velocity,
                                                                  hard=False,
                                                                  **kwargs))

    def make_constraints(self):
        r_P_g = self.get_parameter_as_symbolic_expression('goal_point')
        r_P_c = w.position_of(self.get_fk(self.root_link, self.tip_link))
        # self.add_debug_expr('trans', w.norm(r_P_c))
        self.add_point_goal_constraints(frame_P_goal=r_P_g,
                                        frame_P_current=r_P_c,
                                        reference_velocity=self.reference_velocity,
                                        weight=self.weight)

    def __str__(self):
        s = super(CartesianPosition, self).__str__()
        return '{}/{}/{}'.format(s, self.root_link, self.tip_link)


class CartesianOrientation(Goal):
    def __init__(self, root_link, tip_link, goal_orientation, reference_velocity=None, max_velocity=0.5,
                 weight=WEIGHT_ABOVE_CA, **kwargs):
        super(CartesianOrientation, self).__init__(**kwargs)
        if reference_velocity is None:
            reference_velocity = max_velocity
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_orientation = self.transform_msg(self.root_link, goal_orientation)
        self.reference_velocity = reference_velocity
        self.max_velocity = max_velocity
        self.weight = weight
        # if self.max_velocity is not None:
        #     self.add_constraints_of_goal(RotationVelocityLimit(root_link=root_link,
        #                                                        tip_link=tip_link,
        #                                                        weight=weight,
        #                                                        max_velocity=max_velocity,
        #                                                        hard=False,
        #                                                        **kwargs))

    def make_constraints(self):
        r_R_g = self.get_parameter_as_symbolic_expression('goal_orientation')
        r_R_c = self.get_fk(self.root_link, self.tip_link)
        c_R_r_eval = self.get_fk_evaluated(self.tip_link, self.root_link)
        self.add_rotation_goal_constraints(frame_R_current=r_R_c,
                                           frame_R_goal=r_R_g,
                                           current_R_frame_eval=c_R_r_eval,
                                           reference_velocity=self.reference_velocity,
                                           weight=self.weight)

    def __str__(self):
        s = super(CartesianOrientation, self).__str__()
        return '{}/{}/{}'.format(s, self.root_link, self.tip_link)


class CartesianPositionStraight(Goal):
    def __init__(self, root_link: str, tip_link: str, goal_point: PointStamped, reference_velocity: float = None,
                 max_velocity: float = 0.2, weight: float = WEIGHT_ABOVE_CA, **kwargs):
        super(CartesianPositionStraight, self).__init__(**kwargs)
        if reference_velocity is None:
            reference_velocity = max_velocity
        self.reference_velocity = reference_velocity
        self.max_velocity = max_velocity
        self.weight = weight
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_point = self.transform_msg(self.root_link, goal_point)

    def make_constraints(self):
        root_P_goal = self.get_parameter_as_symbolic_expression('goal_point')
        root_P_tip = w.position_of(self.get_fk(self.root_link, self.tip_link))
        t_T_r = self.get_fk(self.tip_link, self.root_link)
        tip_P_goal = w.dot(t_T_r, root_P_goal)

        # Create rotation matrix, which rotates the tip link frame
        # such that its x-axis shows towards the goal position.
        # The goal frame is called 'a'.
        # Thus, the rotation matrix is called t_R_a.
        tip_P_error = tip_P_goal[:3]
        trans_error = w.norm(tip_P_error)
        # x-axis
        tip_P_intermediate_error = w.save_division(tip_P_error, trans_error)[:3]
        # y- and z-axis
        tip_P_intermediate_y = w.scale(w.Matrix(np.random.random((3,))), 1)
        y = w.cross(tip_P_intermediate_error, tip_P_intermediate_y)
        z = w.cross(tip_P_intermediate_error, y)
        t_R_a = w.Matrix([[tip_P_intermediate_error[0], -z[0], y[0], 0],
                          [tip_P_intermediate_error[1], -z[1], y[1], 0],
                          [tip_P_intermediate_error[2], -z[2], y[2], 0],
                          [0, 0, 0, 1]])
        t_R_a = w.normalize_rotation_matrix(t_R_a)

        # Apply rotation matrix on the fk of the tip link
        a_T_t = w.dot(w.inverse_frame(t_R_a) ,
                      self.get_fk_evaluated(self.tip_link, self.root_link),
                      self.get_fk(self.root_link, self.tip_link))
        expr_p = w.position_of(a_T_t)
        dist = w.norm(root_P_goal - root_P_tip)

        #self.add_debug_vector(self.tip_link + '_P_goal', tip_P_error)
        #self.add_debug_matrix(self.tip_link + '_R_frame', t_R_a)
        #self.add_debug_matrix(self.tip_link + '_T_a', w.inverse_frame(a_T_t))
        #self.add_debug_expr('error', dist)

        self.add_constraint_vector(reference_velocities=[self.reference_velocity] * 3,
                                   lower_errors=[dist, 0, 0],
                                   upper_errors=[dist, 0, 0],
                                   weights=[WEIGHT_ABOVE_CA, WEIGHT_ABOVE_CA * 2, WEIGHT_ABOVE_CA * 2],
                                   expressions=expr_p[:3],
                                   name_suffixes=['{}/x'.format('line'),
                                                  '{}/y'.format('line'),
                                                  '{}/z'.format('line')])

        if self.max_velocity is not None:
            self.add_translational_velocity_limit(frame_P_current=root_P_tip,
                                                  max_velocity=self.max_velocity,
                                                  weight=self.weight)


class CartesianPose(Goal):
    def __init__(self, root_link: str, tip_link: str, goal_pose: PoseStamped, max_linear_velocity: float = 0.1,
                 max_angular_velocity: float = 0.5, weight: float = WEIGHT_ABOVE_CA, **kwargs):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: str, name of the root link of the kin chain
        :param tip_link: str, name of the tip link of the kin chain
        :param goal_pose: PoseStamped as json
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianPose, self).__init__(**kwargs)
        goal_point = PointStamped()
        goal_point.header = goal_pose.header
        goal_point.point = goal_pose.pose.position
        self.add_constraints_of_goal(CartesianPosition(root_link=root_link,
                                                       tip_link=tip_link,
                                                       goal_point=goal_point,
                                                       max_velocity=max_linear_velocity,
                                                       weight=weight,
                                                       **kwargs))
        goal_orientation = QuaternionStamped()
        goal_orientation.header = goal_pose.header
        goal_orientation.quaternion = goal_pose.pose.orientation
        self.add_constraints_of_goal(CartesianOrientation(root_link=root_link,
                                                          tip_link=tip_link,
                                                          goal_orientation=goal_orientation,
                                                          max_velocity=max_angular_velocity,
                                                          weight=weight,
                                                          **kwargs))


class DiffDriveBaseGoal(CartesianPose):

    def __init__(self, root_link: str, tip_link: str, goal_pose: PoseStamped, max_linear_velocity: float = 0.1,
                 max_angular_velocity: float = 0.5, weight: float = WEIGHT_ABOVE_CA, pointing_axis=None, **kwargs):
        super().__init__(root_link, tip_link, goal_pose, max_linear_velocity, max_angular_velocity, weight, **kwargs)
        goal_point = PointStamped()
        goal_point.header = goal_pose.header
        goal_point.point = goal_pose.pose.position
        if pointing_axis is None:
            pointing_axis = Vector3Stamped()
            pointing_axis.header.frame_id = tip_link
            pointing_axis.vector.x = 1
        #TODO handle weights properly
        self.add_constraints_of_goal(PointingDiffDrive(tip_link=tip_link,
                                                       root_link=root_link,
                                                       goal_point=goal_point,
                                                       pointing_axis=pointing_axis,
                                                       max_velocity=max_angular_velocity,
                                                       **kwargs))


class CartesianPoseStraight(Goal):
    def __init__(self, root_link: str, tip_link: str, goal_pose: PoseStamped, max_linear_velocity : float = 0.1,
                 max_angular_velocity: float = 0.5, weight: float = WEIGHT_ABOVE_CA, **kwargs):
        super(CartesianPoseStraight, self).__init__(**kwargs)
        goal_point = PointStamped()
        goal_point.header = goal_pose.header
        goal_point.point = goal_pose.pose.position
        self.add_constraints_of_goal(CartesianPositionStraight(root_link=root_link,
                                                               tip_link=tip_link,
                                                               goal_point=goal_point,
                                                               max_velocity=max_linear_velocity,
                                                               weight=weight,
                                                               **kwargs))
        goal_orientation = QuaternionStamped()
        goal_orientation.header = goal_pose.header
        goal_orientation.quaternion = goal_pose.pose.orientation
        self.add_constraints_of_goal(CartesianOrientation(root_link=root_link,
                                                          tip_link=tip_link,
                                                          goal_orientation=goal_orientation,
                                                          max_velocity=max_angular_velocity,
                                                          weight=weight,
                                                          **kwargs))


class TranslationVelocityLimit(Goal):
    def __init__(self, root_link, tip_link, weight=WEIGHT_ABOVE_CA, max_velocity=0.1, hard=True, **kwargs):
        """
        This goal will limit the cartesian velocity of the tip link relative to root link
        :param root_link: str, root link of the kin chain
        :param tip_link: str, tip link of the kin chain
        :param weight: float, default WEIGHT_ABOVE_CA
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param hard: bool, default True, will turn this into a hard constraint, that will always be satisfied, can could
                                make some goal combination infeasible
        """
        self.root_link = root_link
        self.tip_link = tip_link
        self.hard = hard
        self.weight = weight
        self.max_velocity = max_velocity
        super(TranslationVelocityLimit, self).__init__(**kwargs)

    def make_constraints(self):
        r_P_c = w.position_of(self.get_fk(self.root_link, self.tip_link))
        # self.add_debug_expr('limit', -self.max_velocity)
        if not self.hard:
            self.add_translational_velocity_limit(frame_P_current=r_P_c,
                                                  max_velocity=self.max_velocity,
                                                  weight=self.weight)
        else:
            self.add_translational_velocity_limit(frame_P_current=r_P_c,
                                                  max_velocity=self.max_velocity,
                                                  weight=self.weight,
                                                  max_violation=0)

    def __str__(self):
        s = super(TranslationVelocityLimit, self).__str__()
        return '{}/{}/{}'.format(s, self.root_link, self.tip_link)


class RotationVelocityLimit(Goal):
    def __init__(self, root_link, tip_link, weight=WEIGHT_ABOVE_CA, max_velocity=0.5, hard=True, **kwargs):
        """
        This goal will limit the cartesian velocity of the tip link relative to root link
        :param root_link: str, root link of the kin chain
        :param tip_link: str, tip link of the kin chain
        :param weight: float, default WEIGHT_ABOVE_CA
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param hard: bool, default True, will turn this into a hard constraint, that will always be satisfied, can could
                                make some goal combination infeasible
        """
        self.root_link = root_link
        self.tip_link = tip_link
        self.hard = hard

        self.weight = weight
        self.max_velocity = max_velocity
        super(RotationVelocityLimit, self).__init__(**kwargs)

    def make_constraints(self):
        r_R_c = w.rotation_of(self.get_fk(self.root_link, self.tip_link))
        if self.hard:
            self.add_rotational_velocity_limit(frame_R_current=r_R_c,
                                               max_velocity=self.max_velocity,
                                               weight=self.weight)
        else:
            self.add_rotational_velocity_limit(frame_R_current=r_R_c,
                                               max_velocity=self.max_velocity,
                                               weight=self.weight,
                                               max_violation=0)

    def __str__(self):
        s = super(RotationVelocityLimit, self).__str__()
        return '{}/{}/{}'.format(s, self.root_link, self.tip_link)


class CartesianVelocityLimit(Goal):
    def __init__(self, root_link, tip_link, max_linear_velocity=0.1, max_angular_velocity=0.5, weight=WEIGHT_ABOVE_CA,
                 hard=False, **kwargs):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: str, name of the root link of the kin chain
        :param tip_link: str, name of the tip link of the kin chain
        :param goal: PoseStamped as json
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianVelocityLimit, self).__init__(**kwargs)
        self.add_constraints_of_goal(TranslationVelocityLimit(root_link=root_link,
                                                              tip_link=tip_link,
                                                              max_velocity=max_linear_velocity,
                                                              weight=weight,
                                                              hard=hard,
                                                              **kwargs))
        self.add_constraints_of_goal(RotationVelocityLimit(root_link=root_link,
                                                           tip_link=tip_link,
                                                           max_velocity=max_angular_velocity,
                                                           weight=weight,
                                                           hard=hard,
                                                           **kwargs))


class CartesianPathCarrot(Goal):

    def __init__(self, root_link, tip_link, goal, goals=None, max_linear_velocity=0.1,
                 max_angular_velocity=0.5, max_linear_acceleration=0.1, max_angular_acceleration=0.5,
                 weight=WEIGHT_ABOVE_CA, predict_f = 1.0, start=None, path_length=None,
                 narrow=False, narrow_padding=1.0, **kwargs):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: str, name of the root link of the kin chain
        :param tip_link: str, name of the tip link of the kin chain
        :param goal: PoseStamped as json
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianPathCarrot, self).__init__(**kwargs)
        self.root_link = root_link
        self.tip_link = tip_link
        self.initialized = False

        self.terminal_goal = tf.transform_pose(self.root_link, goal)
        self.weight = weight
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.max_linear_acceleration = max_linear_acceleration
        self.max_angular_acceleration = max_angular_acceleration
        self.predict_f = predict_f
        self.arriving_thresh = 0.2
        self.min_v = self.god_map.get_data(identifier.joint_convergence_threshold)

        if goals is not None and len(goals) != 0:
            self.trajectory_length = len(goals)
            self.setup_goal_params(goals)
            self.initialized = True

    def setup_goal_params(self, goals):
        goal_str = []
        next_goal_strings = []
        self.params_goals = {}

        for i in range(0, len(goals)):
            self.params_goals['goal_' + str(i)] = tf.transform_pose(self.root_link, goals[i])
            goal_str.append('goal_' + str(i))
            if i != 0:
                next_goal_strings.append('goal_' + str(i))

        next_goal_strings.append('goal_' + str(len(goals) - 1))
        self.next_goal_strings = next_goal_strings
        self.goal_strings = goal_str

    def get_goal_expr(self):
        """Will return the next normal with a higher normal time as the normal time calculated from the current position."""
        # todo: clean the mess below
        # Translation Calculation
        # Calculate the normals and normal times with the current robot position and ...
        current_P = w.position_of(self.get_fk(self.root_link, self.tip_link))
        curr_normals, curr_normal_times = self.get_normals(self.goal_strings, self.next_goal_strings, current_P)
        curr_normal_dists = self.get_normal_dists(curr_normals, current_P)
        # ... choose the closest normal with its estimated normal time.
        zero_one_mapping = self.zero_one_mapping_if_equal(curr_normal_dists, self.trajectory_length, w.ca.mmin(curr_normal_dists))
        curr_normal_time = self.select(curr_normal_times, zero_one_mapping)
        # Calculate the normals and normal times with the predicted robot position and ...
        next_P = self.predict()
        next_normals, next_normal_times = self.get_normals(self.goal_strings, self.next_goal_strings, next_P)
        # ... filter the normals out which have a smaller normal time than curr_normal_time.
        zero_one_mapping_n = self.zero_one_mapping_if_greater(next_normal_times, self.trajectory_length, curr_normal_time)
        next_normals_closer_to_goal = next_normals * zero_one_mapping_n
        # After that get the closest normal point relative to the predicted robot position.
        next_normal_dists = self.get_normal_dists(next_normals_closer_to_goal, next_P)
        zero_one_mapping_one = self.zero_one_mapping_if_equal(next_normal_dists, self.trajectory_length,
                                                              w.ca.mmin(next_normal_dists))
        next_normal = self.select(next_normals_closer_to_goal, zero_one_mapping_one)
        # Orientation Calculation
        decimal_of_curr_normal_time = curr_normal_time - w.round_down(curr_normal_time, 0)
        line_starts = []
        line_ends = []
        for i in range(0, self.trajectory_length):
            line_s_q = w.quaternion_from_matrix(
                w.rotation_of(self.get_parameter_as_symbolic_expression([u'params_goals', self.goal_strings[i]])))
            line_starts.append(line_s_q)
            line_e_q = w.quaternion_from_matrix(
                w.rotation_of(self.get_parameter_as_symbolic_expression([u'params_goals', self.next_goal_strings[i]])))
            line_ends.append(line_e_q)
        line_start_q = self.select(w.Matrix(line_starts), zero_one_mapping)
        line_end_q = self.select(w.Matrix(line_ends), zero_one_mapping)
        current_rotation = w.quaternion_slerp(line_start_q, line_end_q, decimal_of_curr_normal_time)
        next_rotation = w.rotation_matrix_from_quaternion(current_rotation[0],
                                                          current_rotation[1],
                                                          current_rotation[2],
                                                          current_rotation[3])
        return next_normal, next_rotation

    def predict(self):
        v = self.get_fk_velocity(self.root_link, self.tip_link)[0:3]
        v_p = w.save_division(v, w.norm(v), 0) * self.get_velocity() * self.predict_f
        p = w.position_of(self.get_fk(self.root_link, self.tip_link))
        s = self.get_sampling_period_symbol()
        n_p = p[0:3] + v_p * s
        return w.point3(n_p[0], n_p[1], n_p[2])

    def get_normal_time(self, n, a, b, ps):
        """
        Will return the normal time for a given normal point n between the start point a and b.
        First the normal time depends on the place in the trajectory. If a (the start point of the given
        trajectory part) is at the end of the trajectory the time is higher. The normal time can therefore
        be in [0, len(self.trajectory_length)]. If n is not a or b, the normalized distance from a to n will
        be added on the normal time. This results in the following formulation:
        normal_time = trajectory[a].index() + norm(n-a)/norm(b-a)
        """
        m = self.zero_one_mapping_if_equal(w.Matrix(ps), self.trajectory_length, w.ca.transpose(a))
        g_i = self.select(w.Matrix([i for i in range(0, self.trajectory_length)]), m)
        n_t = w.save_division(w.norm(n - a), w.norm(b - a))
        return g_i + n_t

    def get_normal(self, p, a, b):
        """:rtype: w.point3"""
        ap = p - a
        ab = b - a
        ab = w.save_division(ab, w.norm(ab), 0)
        ab = ab * (w.dot(w.ca.transpose(ap), ab))
        normal = a + ab
        normal = w.if_less(normal[0], w.min(a[0], b[0]), b, normal)
        normal = w.if_greater(normal[0], w.max(a[0], b[0]), b, normal)
        return normal

    def get_normals(self, goal_strings, next_goal_strings, pos):

        trajectory_len = self.trajectory_length

        ps = []
        for i in range(0, self.trajectory_length):
            ps.append(w.position_of(self.get_parameter_as_symbolic_expression([u'params_goals', self.goal_strings[i]])))

        normals = []
        normal_times = []
        for i in range(0, trajectory_len):
            a = w.position_of(self.get_parameter_as_symbolic_expression([u'params_goals', goal_strings[i]]))
            b = w.position_of(self.get_parameter_as_symbolic_expression([u'params_goals', next_goal_strings[i]]))
            n = self.get_normal(pos, a, b)
            n_t = self.get_normal_time(n, a, b, ps)
            normals.append(n)
            normal_times.append(n_t)

        return w.Matrix(normals), w.Matrix(normal_times)

    def get_normal_dists(self, normals, pos):

        trajectory_len = self.trajectory_length
        normal_dist_funs = []

        for i in range(0, trajectory_len):
            normal_dist_funs.append(w.manhattan_norm(w.ca.transpose(normals[i,:]), pos))

        return w.Matrix(normal_dist_funs)

    def zero_one_mapping_if_equal(self, l, l_len, elem):
        """
        Will take 2d array l and will compare elementwise each 1d entry of l with the given array elem.
        The compare function is here equal(). Because of the elementwise comparison, each positive
        comparison will be summed with 1/l.shape[1]. If the sum is 1 all elements were equal.
        """

        zeros_and_ones = []

        for i in range(0, l_len):
            zeros_and_ones.append(w.if_eq(1, w.sum(w.if_eq(l[i,:], elem, 1/l.shape[1], 0)), 1, 0))

        return w.Matrix(zeros_and_ones)

    def zero_one_mapping_if_greater(self, l, l_len, elem):
        """
        Will take 2d array l and will compare elementwise each 1d entry of l with the given array elem.
        The compare function is here greater(). Because of the elementwise comparison, each positive
        comparison will be summed with the 1/l.shape[1]. If the sum is 1 all entries in l were greater.
        """

        zeros_and_ones = []

        for i in range(0, l_len):
            zeros_and_ones.append(w.if_eq(1, w.sum(w.if_greater(l[i,:], elem, 1/l.shape[1], 0)), 1, 0))

        return w.Matrix(zeros_and_ones)

    def select(self, arr, zero_one_mapping):

        arr_and_zeros = arr * zero_one_mapping
        selected = w.sum_row(arr_and_zeros)
        return w.ca.transpose(selected)

    def make_constraints(self):

        if self.initialized:
            goal_translation, goal_orientation = self.get_goal_expr()

            #self.add_debug_vector("debugGoal", goal_translation)
            #self.add_debug_vector("debugCurrentX", w.position_of(self.get_fk(self.root_link, self.tip_link)))
            #self.add_debug_vector("debugNext", self.predict())

            self.minimize_position(goal_translation, self.get_dyn_weight())
            self.minimize_rotation(goal_orientation, self.weight)

    def get_closest_traj_point(self):

        current_pose = self.get_fk(self.root_link, self.tip_link)
        trajectory_len = self.trajectory_length
        dists_l = []
        inds_l = []

        for i in range(0, trajectory_len):
            n = w.norm(w.position_of(current_pose - self.get_parameter_as_symbolic_expression([u'params_goals', self.goal_strings[i]])))
            dists_l.append(n)
            inds_l.append(i)

        dists = w.Matrix(dists_l)
        inds = w.Matrix(inds_l)
        dist_min = w.ca.mmin(dists)
        mapping = self.zero_one_mapping_if_equal(dists, trajectory_len, dist_min)
        return w.sum_row(inds * mapping)

    def get_dyn_weight(self):
        weight = self.get_parameter_as_symbolic_expression(u'weight')
        dyn_weight = weight - weight * self.get_traversed_trajectory_mult()
        return dyn_weight

    def get_traversed_trajectory_mult(self):
        """
        0 means the traversed trajectory is empty
        1 means we traversed the whole trajectory
        """
        traj_point = self.get_closest_traj_point() + 1
        return traj_point/self.trajectory_length

    def get_velocity(self):
        """
        :rtype: float
        :returns: float in range(0,1)
        """
        terminal_goal = self.get_parameter_as_symbolic_expression(u'terminal_goal')
        dis_to_goal = w.norm(w.position_of(self.get_fk(self.root_link, self.tip_link) - terminal_goal))
        distance_thresh = self.arriving_thresh
        v = w.if_less(dis_to_goal, distance_thresh,
                      self.max_linear_velocity * dis_to_goal/distance_thresh,
                      self.max_linear_velocity)
        return v

    def minimize_position(self, goal, weight):
        max_velocity = self.get_velocity()

        self.add_point_goal_constraints(frame_P_current=w.position_of(self.get_fk(self.root_link, self.tip_link)),
                                        frame_P_goal=goal,
                                        reference_velocity=max_velocity,
                                        weight=weight,
                                        tip_link=self.tip_link,
                                        root_link=self.root_link,
                                        name_suffix=u'goal_pos')

    def minimize_rotation(self, goal, weight):
        max_velocity = self.max_angular_velocity
        r_R_g = goal
        r_R_c = self.get_fk(self.root_link, self.tip_link)
        c_R_r_eval = self.get_fk_evaluated(self.tip_link, self.root_link)

        self.add_rotation_goal_constraints(frame_R_current=r_R_c,
                                           frame_R_goal=r_R_g,
                                           current_R_frame_eval=c_R_r_eval,
                                           reference_velocity=max_velocity,
                                           weight=weight, name_suffix=u'goal_rot')

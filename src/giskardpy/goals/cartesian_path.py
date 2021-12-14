import time

import numpy as np

from giskardpy import identifier
import giskardpy.utils.tfwrapper as tf
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE, WEIGHT_BELOW_CA
import giskardpy.casadi_wrapper as w


class CartesianPathCarrot(Goal):

    def __init__(self, root_link, tip_link, goals, max_linear_velocity=0.1,
                 max_angular_velocity=0.5, max_linear_acceleration=0.1, max_angular_acceleration=0.5,
                 weight=WEIGHT_ABOVE_CA, ignore_trajectory_orientation=False, **kwargs):
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
        self.trajectory_length = len(goals)

        self.terminal_goal = tf.transform_pose(self.root_link, goals[-1])
        self.weight = weight
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.max_linear_acceleration = max_linear_acceleration
        self.max_angular_acceleration = max_angular_acceleration
        self.ignore_trajectory_orientation = ignore_trajectory_orientation

        self.setup_goal_params(goals)

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
        curr_normals, curr_normal_times = self.get_normals(self.goal_strings, self.next_goal_strings,
                                                           w.position_of(self.get_fk(self.root_link, self.tip_link)))
        curr_normal_dists = self.get_normal_dists(curr_normals)
        # ... choose the closest normal with its estimated normal time.
        zero_one_mapping = self.zero_one_mapping_if_equal(curr_normal_dists, self.trajectory_length, w.ca.mmin(curr_normal_dists))
        curr_normal_time = self.select(curr_normal_times, zero_one_mapping)
        # Calculate the normals and normal times with the predicted robot position and ...
        next_normals, next_normal_times = self.get_normals(self.goal_strings, self.next_goal_strings, self.predict())
        # ... filter the normals out which have a smaller normal time than curr_normal_time.
        zero_one_mapping_n = self.zero_one_mapping_if_greater(next_normal_times, self.trajectory_length, curr_normal_time)
        next_normals_closer_to_goal = next_normals * zero_one_mapping_n
        # After that get the closest normal point relative to the predicted robot position.
        next_normal_dists = self.get_normal_dists(next_normals_closer_to_goal)
        zero_one_mapping_one = self.zero_one_mapping_if_equal(next_normal_dists, self.trajectory_length,
                                                              w.ca.mmin(next_normal_dists))
        next_normal = self.select(next_normals_closer_to_goal, zero_one_mapping_one)
        # Orientation Calculation
        if not self.ignore_trajectory_orientation:
            decimal_of_curr_normal_time = curr_normal_time - w.round_down(curr_normal_time, 0)
            line_starts = []
            line_ends = []
            for i in range(0, self.trajectory_length):
                line_s_q = w.quaternion_from_matrix(w.rotation_of(self.get_parameter_as_symbolic_expression([u'params_goals', self.goal_strings[i]])))
                line_starts.append(line_s_q)
                line_e_q = w.quaternion_from_matrix(w.rotation_of(self.get_parameter_as_symbolic_expression([u'params_goals', self.next_goal_strings[i]])))
                line_ends.append(line_e_q)
            line_start_q = self.select(w.Matrix(line_starts), zero_one_mapping)
            line_end_q = self.select(w.Matrix(line_ends), zero_one_mapping)
            current_rotation = w.quaternion_slerp(line_start_q, line_end_q, decimal_of_curr_normal_time)
            next_rotation = w.rotation_matrix_from_quaternion(current_rotation[0], current_rotation[1], current_rotation[2], current_rotation[3])
        else:
            next_rotation = w.rotation_of(self.get_parameter_as_symbolic_expression([u'params_goals', self.goal_strings[-1]]))
        return next_normal, next_rotation

    def predict(self):
        v = self.get_fk_velocity(self.root_link, self.tip_link)[0:3]
        v_p = w.save_division(v, w.norm(v), 0) * self.max_linear_velocity * 10 #todo: make parameter for num
        p = w.position_of(self.get_fk(self.root_link, self.tip_link))
        s = self.get_sampling_period_symbol()
        n_p = p[0:3] + v_p * s
        return w.point3(n_p[0], n_p[1], n_p[2])

    def get_normal_time(self, n, a, b):
        """
        Will return the normal time for a given normal point n between the start point a and b.
        First the normal time depends on the place in the trajectory. If a (the start point of the given
        trajectory part) is at the end of the trajectory the time is higher. The normal time can therefore
        be in [0, len(self.trajectory_length)]. If n is not a or b, the normalized distance from a to n will
        be added on the normal time. This results in the following formulation:
        normal_time = trajectory[a].index() + norm(n-a)/norm(b-a)
        """
        ps = []
        for i in range(0, self.trajectory_length):
            ps.append(w.position_of(self.get_parameter_as_symbolic_expression([u'params_goals', self.goal_strings[i]])))
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
        normals = []
        normal_times = []

        for i in range(0, trajectory_len):
            a = w.position_of(self.get_parameter_as_symbolic_expression([u'params_goals', goal_strings[i]]))
            b = w.position_of(self.get_parameter_as_symbolic_expression([u'params_goals', next_goal_strings[i]]))
            n = self.get_normal(pos, a, b)
            n_t = self.get_normal_time(n, a, b)
            normals.append(n)
            normal_times.append(n_t)

        return w.Matrix(normals), w.Matrix(normal_times)

    def get_normal_dists(self, normals):

        trajectory_len = self.trajectory_length
        next_pos = self.predict()
        normal_dist_funs = []

        for i in range(0, trajectory_len):
            normal_dist_funs.append(w.norm(w.ca.transpose(normals[i,:]) - next_pos))

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

        goal_translation, goal_orientation = self.get_goal_expr()

        self.add_debug_vector("debugGoal", goal_translation)
        self.add_debug_vector("debugCurrentX", w.position_of(self.get_fk(self.root_link, self.tip_link)))
        self.add_debug_vector("debugNext", self.predict())

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

    def get_terminal_goal_weight_mult(self):
        """
        behaves pretty shitty

        0 means the goal is far away, 1 means the goal is close
        :rtype: float
        :returns: float in range(0,1)
        """
        terminal_goal = self.get_parameter_as_symbolic_expression(u'terminal_goal')
        dis_to_goal = w.norm(w.position_of(self.get_fk(self.root_link, self.tip_link) - terminal_goal))
        distance_thresh = 1.0
        return w.if_less(dis_to_goal, distance_thresh,
                         0.5 * (distance_thresh - dis_to_goal) + 0.5, # add 0.5 as starting point from below:
                         distance_thresh / (2 * dis_to_goal)) # if dis_to_goal == distance_thresh, then
                                                              # distance_thresh / 2 * dis_to_goal == 0.5.

    def minimize_position(self, goal, weight):
        max_velocity = self.max_linear_velocity

        self.add_point_goal_constraints(frame_P_current=w.position_of(self.get_fk(self.root_link, self.tip_link)),
                                        frame_P_goal=goal,
                                        reference_velocity=max_velocity,
                                        weight=weight, name_suffix=u'goal_pos')

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


class CartesianPathSplineCarrot(Goal):
    LUT_x_sym = u'LUT_x'
    LUT_y_sym = u'LUT_y'
    terminal_goal=u'terminal_goal'
    weight = u'weight'
    max_linear_velocity = u'max_linear_velocity'
    max_angular_velocity = u'max_angular_velocity'
    max_linear_acceleration = u'max_linear_acceleration'
    max_angular_acceleration = u'max_angular_acceleration'

    def __init__(self, god_map, root_link, tip_link, goals, max_linear_velocity=0.1,
                 max_angular_velocity=0.5, max_linear_acceleration=0.1, max_angular_acceleration=0.5,
                 weight=WEIGHT_ABOVE_CA, goal_constraint=False):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: str, name of the root link of the kin chain
        :param tip_link: str, name of the tip link of the kin chain
        :param goal: PoseStamped as json
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianPathSplineCarrot, self).__init__(god_map)
        self.constraints = []
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_constraint = goal_constraint
        self.trajectory_length = len(goals)
        self.calc_splines(goals)

        params = {
            self.LUT_x_sym: self.LUT_x_f,
            self.LUT_y_sym: self.LUT_y_f,
            self.terminal_goal: self.parse_and_transform_PoseStamped(goals[-1], root_link),
            self.weight: weight,
            self.max_linear_velocity: max_linear_velocity,
            self.max_angular_velocity: max_angular_velocity,
            self.max_linear_acceleration: max_linear_acceleration,
            self.max_angular_acceleration: max_angular_acceleration
        }

        goal_str = []
        for i in range(0, len(goals)):
            params['goal_' + str(i)] = self.parse_and_transform_PoseStamped(goals[i], root_link)
            goal_str.append('goal_' + str(i))#w.ca.SX.sym('goal_' + str(i)))

        self.goal_strings = goal_str#w.Matrix(goal_str)
        self.save_params_on_god_map(params)

    def LUT_x_f(self, ts):
        t = self.get_god_map().unsafe_get_data(identifier.time)#fixme:evtl berechnen?
        return self.LUT_x(t*ts)

    def LUT_y_f(self, ts):
        t = self.get_god_map().unsafe_get_data(identifier.time)
        return self.LUT_y(t*ts)

    def calc_splines(self, goals):
        goals_rs = [self.parse_and_transform_PoseStamped(goal, self.root_link) for goal in goals]
        xs = [0 for i in range(0, len(goals_rs))]
        ys = [0 for i in range(0, len(goals_rs))]
        for i, goal in enumerate(goals_rs):
            xs[i] = goal.pose.position.x
            ys[i] = goal.pose.position.y
        t = np.linspace(0, len(goals_rs) - 1, len(goals_rs)).astype(int).tolist()
        LUT_x = w.ca.interpolant("LUT_x", "bspline", [t], xs)
        LUT_y = w.ca.interpolant("LUT_y", "bspline", [t], ys)
        self.LUT_x = LUT_x
        self.LUT_y = LUT_y

    def get_closest_index(self):
        """
        :rtype: int
        """

        trajectory_len = self.trajectory_length
        goal_strings = self.goal_strings

        distances = []
        for i in range(0, trajectory_len):
            distances.append(self.distance_to_goal(goal_strings[i]))
        min_dis = w.ca.mmin(w.Matrix(distances))

        min_inds = []
        for i in range(0, trajectory_len):
            min_inds.append(w.if_eq(distances[i] - min_dis, 0, i, trajectory_len + 1))

        return w.ca.mmin(w.Matrix(min_inds))

    def zero_one_mapping_if_equal(self, l, l_len, elem):

        zeros_and_ones = []

        for i in range(0, l_len):
            zeros_and_ones.append(w.if_eq(l[i], elem, 1, 0))

        return w.Matrix(zeros_and_ones)

    def select(self, arr, zero_one_mapping):

        arr_and_zeros = arr * zero_one_mapping
        selected = w.sum_row(arr_and_zeros)
        return w.ca.transpose(selected)

    def get_closer_carrot_cart(self, max_dist):
        carrots_l = []
        carrots_dist_l = []
        fake_sampling_times = [0.081, 0.071, 0.061, 0.051, 0.041, 0.031, 0.021, 0.011]
        for st in fake_sampling_times:
            p = w.point3(self.get_god_map().to_symbol(self.get_identifier() + [self.LUT_x_sym, (st,)]),
                         self.get_god_map().to_symbol(self.get_identifier() + [self.LUT_y_sym, (st,)]),
                         0)
            p_dist = w.norm(p - w.position_of(self.get_fk(self.root_link, self.tip_link)))
            close_to_max_dist = w.if_greater_zero(p_dist - max_dist, -max_dist, p_dist - max_dist)
            carrots_l.append(p)
            carrots_dist_l.append(close_to_max_dist)
        carrots = w.Matrix(carrots_l)
        carrots_dist = w.Matrix(carrots_dist_l)
        max_close_to_max_dist = w.ca.mmax(carrots_dist)
        mapping = self.zero_one_mapping_if_equal(carrots_dist, 8, max_close_to_max_dist)
        return self.select(carrots, mapping)

    def get_carrot(self):
        closest_index = self.get_closest_index()
        closest_time = closest_index
        #goal_time = w.if_greater_eq(closest_time + 0.5, self.trajectory_length,
        #                            self.trajectory_length, closest_time + 0.5)
        x = self.get_god_map().to_symbol(self.get_identifier() + [self.LUT_x_sym, (0.101,)])
        y = self.get_god_map().to_symbol(self.get_identifier() + [self.LUT_y_sym, (0.101,)])
        p = w.point3(x, y, 0)
        self.add_debug_constraint("dist_carrot", w.norm(p - w.position_of(self.get_fk(self.root_link, self.tip_link))))
        return p#self.get_closer_carrot_cart(0.5)

    def make_constraints(self):
        goal_translation = self.get_carrot()
        time = self.get_god_map().get_data(identifier.time)
        weight = self.get_input_float(self.weight)
        trajectory_len = self.get_input_float(self.trajectory_length)
        #dis_to_goal = self.distance_to_goal(self.terminal_goal)
        dyn_weight = weight - weight * w.if_greater(time/trajectory_len, 1, 1, time/trajectory_len)
                                        #0.3*w.if_less(dis_to_goal, 0.2, 1, w.if_less(dis_to_goal, 1, 1/30, 1/dis_to_goal)))
        self.add_debug_constraint("debugGoalX", goal_translation[0])
        self.add_debug_constraint("debugGoalY", goal_translation[1])
        self.add_debug_constraint("debugGoalZ", goal_translation[2])
        self.minimize_position(goal_translation, WEIGHT_ABOVE_CA)

    def distance_to_goal(self, p):
        return w.norm(w.position_of(self.get_input_PoseStamped(p) - self.get_fk(self.root_link, self.tip_link)))

    def minimize_pose(self, goal, weight):
        self.minimize_position(w.position_of(goal), weight)
        self.minimize_rotation(w.rotation_of(goal), weight)

    def minimize_position(self, goal, weight):
        max_velocity = self.get_input_float(self.max_linear_velocity)
        max_acceleration = self.get_input_float(self.max_linear_acceleration)

        self.add_minimize_position_constraints(goal, max_velocity, max_acceleration, self.root_link, self.tip_link,
                                               self.goal_constraint, weight, prefix='goal')

    def minimize_rotation(self, goal, weight):
        max_velocity = self.get_input_float(self.max_angular_velocity)
        max_acceleration = self.get_input_float(self.max_angular_acceleration)

        self.add_minimize_rotation_constraints(goal, self.root_link, self.tip_link, max_velocity, weight,
                                               self.goal_constraint, prefix='goal')

class CartesianPathError(Goal):
    get_next_py_f = u'get_next_py_f'
    goal = u'goal'
    goal_time = u'goal_time'
    weight = u'weight'
    max_linear_velocity = u'max_linear_velocity'
    max_angular_velocity = u'max_angular_velocity'
    max_linear_acceleration = u'max_linear_acceleration'
    max_angular_acceleration = u'max_angular_acceleration'

    def __init__(self, god_map, root_link, tip_link, goals, max_linear_velocity=0.1,
                 max_angular_velocity=0.5, max_linear_acceleration=0.1, max_angular_acceleration=0.5,
                 weight=WEIGHT_ABOVE_CA, goal_constraint=False):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: str, name of the root link of the kin chain
        :param tip_link: str, name of the tip link of the kin chain
        :param goal: PoseStamped as json
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianPathError, self).__init__(god_map)
        self.constraints = []
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_constraint = goal_constraint
        self.trajectory_length = len(goals)

        params = {
            #self.goal: self.goals_rs[0],
            #self.goal_time: 0,
            self.weight: weight,
            self.max_linear_velocity: max_linear_velocity,
            self.max_angular_velocity: max_angular_velocity,
            self.max_linear_acceleration: max_linear_acceleration,
            self.max_angular_acceleration: max_angular_acceleration
        }

        goal_str = []
        next_goal_strings = []
        for i in range(0, len(goals)):
            params['goal_' + str(i)] = self.parse_and_transform_PoseStamped(goals[i], root_link)
            goal_str.append(w.ca.SX.sym('goal_' + str(i)))
            if i != 0:
                next_goal_strings.append(w.ca.SX.sym('goal_' + str(i)))

        next_goal_strings.append(w.ca.SX.sym('goal_' + str(len(goals) - 1)))
        self.next_goal_strings = w.Matrix(next_goal_strings)
        self.goal_strings = w.Matrix(goal_str)
        self.save_params_on_god_map(params)

    # def make_constraints(self):
    #    self.get_god_map().to_symbol(self.get_identifier() + [self.get_next_py_f, tuple()])
    #    self.minimize_pose(self.goal, WEIGHT_COLLISION_AVOIDANCE)

    #def calc_splines(self):
    #    xs = np.zeros(len(self.goals_rs))
    #    ys = np.array(len(self.goals_rs))
    #    for i, goal in enumerate(self.goals_rs):
    #        xs[i] = goal.pose.position.x
    #        ys[i] = goal.pose.position.y
    #    self.LUT_x = w.ca.interpolant("self.LUT_x", "bspline",
    #                                    np.linspace(0, len(self.goals_rs), len(self.goals_rs) - 1), xs)
    #    self.LUT_y = w.ca.interpolant("self.LUT_y", "bspline",
    #                                    np.linspace(0, len(self.goals_rs), len(self.goals_rs) - 1), ys)
        # tck, u = interpolate.splprep([xs, ys], s=0.0)
        # x_i, y_i = interpolate.splev(np.linspace(0, 1, 100), tck)
        # self.interpolated_path = np.array([x_i, y_i])

    #def get_error(self, p, v_time):
    #    return w.ca.sumsqr(w.position_of(p)[0] - self.LUT_x(v_time * time)) + \
    #           w.ca.sumsqr(w.position_of(p)[1] - self.LUT_y(v_time * time))

    def get_error_to_goal(self, p):
        #next_index = w.if_greater(index + 1, self.trajectory_length, self.trajectory_length, index + 1)
        next_goal = w.position_of(self.get_input_PoseStamped(self.get_next_goal()))
        return w.ca.sumsqr(w.position_of(p)[0] - next_goal[0]) + \
               w.ca.sumsqr(w.position_of(p)[1] - next_goal[1])

    def get_closest_index(self):
        """
        :rtype: int
        """

        trajectory_len = self.trajectory_length
        goal_strings = self.goal_strings

        x = w.ca.SX.sym('x', 1)
        f = w.ca.Function('f', [x], [self.distance_to_goal(x)])
        f_map = f.map(trajectory_len)
        distances = f_map(goal_strings)
        min_dis = w.ca.mmin(distances)

        d = w.ca.SX.sym('d', 1)
        i = w.ca.SX.sym('i', 1)
        g = w.ca.Function('g', [d, i], [w.if_eq(d - min_dis, 0, i, trajectory_len + 1)])  # FIXME: kinda retarded
        g_map = g.map(trajectory_len)
        min_index = w.ca.mmin(g_map(distances, w.ca.linspace(0, trajectory_len - 1, trajectory_len)))

        return min_index

    def get_next_goal(self):
        """
        :rtype: int
        """

        trajectory_len = self.trajectory_length
        goal_strings = self.goal_strings

        x = w.ca.SX.sym('x', 1)
        f = w.ca.Function('f', [x], [self.distance_to_goal(x)])
        f_map = f.map(trajectory_len)
        distances = f_map(goal_strings)
        min_dis = w.ca.mmin(distances)

        d = w.ca.SX.sym('d', 1)
        i = w.ca.SX.sym('i', 1)
        g = w.ca.Function('g', [d, i], [w.if_eq(d - min_dis, 0, i, trajectory_len + 1)])  # FIXME: kinda retarded
        g_map = g.map(trajectory_len)
        goal_str = w.ca.mmin(g_map(distances, self.next_goal_strings))

        return goal_str

    def get_error(self):
        closest_index = self.get_closest_index()
        return self.get_error_to_goal(self.get_fk(self.root_link, self.tip_link), closest_index)

    #def make_constraints(self):
    #    max_velocity = self.get_input_float(self.max_linear_velocity) #w.Min(self.get_input_float(self.max_linear_velocity),
    #                         #self.get_robot().get_joint_velocity_limit_expr(self.tip_link))
    #    weight = self.get_input_float(self.weight)
#
    #    self.add_debug_constraint("debugGoali", self.get_input_float(self.goal_time))
    #    err = self.get_error()
    #    capped_err = self.limit_velocity(err, max_velocity)
    #    weight = self.normalize_weight(max_velocity, weight)
#
    #    self.add_constraint('',
    #                        lower=capped_err,
    #                        upper=capped_err,
    #                        weight=weight,
    #                        expression=err,
    #                        goal_constraint=self.goal_constraint)

    def make_constraints(self):
        goal_str = self.get_next_goal()
        self.add_debug_constraint("debugGoali", self.get_input_float(self.goal_time))
        self.minimize_position(self.get_input_PoseStamped(goal_str), WEIGHT_COLLISION_AVOIDANCE)

    # def get_closest_index_py(self):
    #    goal_dists = list(map(lambda ps: self.distance_to_goal_py(ps), self.goals_rs))
    #    return goal_dists.index(min(goal_dists))

    #def update_goal_time(self):
    #    goal_time = self.get_input_float(self.goal_time)
    #    next_goal_time = w.if_greater(goal_time + 0.05, 1.0, 1.0, goal_time + 0.05)
    #    new_goal_time = w.if_less(self.distance_to_goal(self.get_pose(goal_time)),
    #                              self.distance_to_goal(self.get_pose(next_goal_time)),
    #                              goal_time, next_goal_time)
    #    self.get_god_map().set_data(self.get_identifier() + [self.goal_time], new_goal_time)

    #def get_next_goal(self):
    #    self.update_goal_time()
    #    goal_time = self.get_input_float(self.goal_time)
    #    next_goal_time = w.if_greater(goal_time + 0.05, 1.0, 1.0, goal_time + 0.05)
    #    return self.get_pose(next_goal_time)

    def distance_to_goal(self, p):
        return w.norm(w.position_of(self.get_input_PoseStamped(p) - self.get_fk(self.root_link, self.tip_link)))

    def minimize_pose(self, goal, weight):
        self.minimize_position(goal, weight)
        self.minimize_rotation(goal, weight)

    def minimize_position(self, goal, weight):
        r_P_g = w.position_of(goal)
        max_velocity = self.get_input_float(self.max_linear_velocity)
        max_acceleration = self.get_input_float(self.max_linear_acceleration)

        self.add_minimize_position_constraints(r_P_g, max_velocity, max_acceleration, self.root_link, self.tip_link,
                                               self.goal_constraint, weight, prefix='goal')

    def minimize_rotation(self, goal, weight):
        r_R_g = w.rotation_of(goal)
        max_velocity = self.get_input_float(self.max_angular_velocity)
        max_acceleration = self.get_input_float(self.max_angular_acceleration)

        self.add_minimize_rotation_constraints(r_R_g, self.root_link, self.tip_link, max_velocity, weight,
                                               self.goal_constraint, prefix='goal')

class CartesianPathG(Goal):
    weight = u'weight'
    max_linear_velocity = u'max_linear_velocity'
    max_angular_velocity = u'max_angular_velocity'
    max_linear_acceleration = u'max_linear_acceleration'
    max_angular_acceleration = u'max_angular_acceleration'

    def __init__(self, god_map, root_link, tip_link, goals, max_linear_velocity=0.1,
                 max_angular_velocity=0.5, max_linear_acceleration=0.1, max_angular_acceleration=0.5,
                 weight=WEIGHT_ABOVE_CA, goal_constraint=False):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: str, name of the root link of the kin chain
        :param tip_link: str, name of the tip link of the kin chain
        :param goal: PoseStamped as json
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianPathG, self).__init__(god_map)
        self.constraints = []
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_constraint = goal_constraint
        self.trajectory_length = len(goals)

        params = {
            self.weight: weight,
            self.max_linear_velocity: max_linear_velocity,
            self.max_angular_velocity: max_angular_velocity,
            self.max_linear_acceleration: max_linear_acceleration,
            self.max_angular_acceleration: max_angular_acceleration
        }

        goal_str = []
        next_goal_strings = []
        debug_goals = []
        for i in range(0, len(goals)):
            params['goal_' + str(i)] = self.parse_and_transform_PoseStamped(goals[i], root_link)
            goal_str.append(w.ca.SX.sym('goal_' + str(i)))
            debug_goals.append(self.parse_and_transform_PoseStamped(goals[i], root_link))
            if i != 0:
                next_goal_strings.append(w.ca.SX.sym('goal_' + str(i)))

        next_goal_strings.append(w.ca.SX.sym('goal_' + str(len(goals) - 1)))
        self.next_goal_strings = w.Matrix(next_goal_strings)
        self.goal_strings = w.Matrix(goal_str)
        self.debug_goals = debug_goals
        self.save_params_on_god_map(params)

    def get_min_distance(self):
        trajectory_len = self.trajectory_length
        goal_strings = self.goal_strings

        x = w.ca.SX.sym('x', 1)
        f = w.ca.Function('f', [x], [self.distance_to_goal(x)])
        f_map = f.map(trajectory_len)
        distances = f_map(goal_strings)
        return w.ca.mmin(distances), distances

    def get_next_index(self):
        """
        :rtype: int
        """

        min_dis, distances = self.get_min_distance()
        d = w.ca.SX.sym('d', 1)
        i = w.ca.SX.sym('i', 1)
        g = w.ca.Function('g', [d, i], [w.if_eq(d - min_dis, 0, i, self.trajectory_length + 1)])  # FIXME: kinda retarded
        g_map = g.map(self.trajectory_length)
        min_index = w.ca.mmin(g_map(distances, w.ca.linspace(0, self.trajectory_length - 1, self.trajectory_length)))

        return w.if_greater(min_index + 1, self.trajectory_length, self.trajectory_length, min_index + 1)

    def get_next_goal(self):
        """
        :rtype: int
        """
        return self.next_goal_strings[w.ca.evalf(self.get_next_index())]

    def make_constraints(self):
        goal_str = self.get_next_goal()
        d, _ = self.get_min_distance()
        p = w.position_of(self.get_fk(self.root_link, self.tip_link))
        self.add_debug_constraint("debugdist", d)
        self.add_debug_constraint("debugCurrentX", p[0])
        self.add_debug_constraint("debugCurrentY", p[1])
        self.minimize_position(self.get_input_PoseStamped(goal_str), WEIGHT_COLLISION_AVOIDANCE)

    def distance_to_goal(self, p):
        return w.norm(w.position_of(self.get_input_PoseStamped(p) - self.get_fk_evaluated(self.root_link, self.tip_link)))

    def minimize_pose(self, goal, weight):
        self.minimize_position(goal, weight)
        self.minimize_rotation(goal, weight)

    def minimize_position(self, goal, weight):
        r_P_g = w.position_of(goal)
        max_velocity = self.get_input_float(self.max_linear_velocity)
        max_acceleration = self.get_input_float(self.max_linear_acceleration)

        self.add_minimize_position_constraints(r_P_g, max_velocity, max_acceleration, self.root_link, self.tip_link,
                                               self.goal_constraint, weight, prefix='goal')

    def minimize_rotation(self, goal, weight):
        r_R_g = w.rotation_of(goal)
        max_velocity = self.get_input_float(self.max_angular_velocity)
        max_acceleration = self.get_input_float(self.max_angular_acceleration)

        self.add_minimize_rotation_constraints(r_R_g, self.root_link, self.tip_link, max_velocity, weight,
                                               self.goal_constraint, prefix='goal')

class CartesianPathPy(Goal):
    get_weight_py_f = u'get_weight_py_f'
    weight = u'weight'
    max_linear_velocity = u'max_linear_velocity'
    max_angular_velocity = u'max_angular_velocity'
    max_linear_acceleration = u'max_linear_acceleration'
    max_angular_acceleration = u'max_angular_acceleration'

    def __init__(self, god_map, root_link, tip_link, goals, max_linear_velocity=0.1,
                 max_angular_velocity=0.5, max_linear_acceleration=0.1, max_angular_acceleration=0.5,
                 weight=WEIGHT_ABOVE_CA, goal_constraint=False):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: str, name of the root link of the kin chain
        :param tip_link: str, name of the tip link of the kin chain
        :param goal: PoseStamped as json
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianPathPy, self).__init__(god_map)
        self.constraints = []
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_constraint = goal_constraint
        self.goals_times = np.zeros(len(goals))
        self.goals_times[0] = 1.0
        self.goals_rs = [self.parse_and_transform_PoseStamped(goal, root_link) for goal in goals]
        self.robot = self.get_robot()

        params = {
            self.get_weight_py_f: self.get_weight_py,
            self.weight: weight,
            self.max_linear_velocity: max_linear_velocity,
            self.max_angular_velocity: max_angular_velocity,
            self.max_linear_acceleration: max_linear_acceleration,
            self.max_angular_acceleration: max_angular_acceleration
        }
        for i in range(0, len(goals)):
            params['goal_' + str(i)] = self.parse_and_transform_PoseStamped(goals[i], root_link)

        self.save_params_on_god_map(params)

    # def make_constraints(self):
    #    for constraint in self.constraints:
    #        self.soft_constraints.update(constraint.get_constraints())

    # @profile
    def update_goal_times(self):
        time_in_secs = time.time()

        for i, goal_time in enumerate(self.goals_times):
            if i == 0:
                continue
            if goal_time == 0:
                if self.goal_reached_py(self.goals_rs[i - 1]) == 1:
                    self.goals_times[i] = time_in_secs
                else:
                    self.goals_times[i] = 0

    def make_constraints(self):

        time = self.get_god_map().to_symbol(identifier.time)
        time_in_secs = self.get_input_sampling_period() * time
        # path = [self.goal_a, self.goal_b, self.goal_c, self.goal_d]
        #
        # # weight in [0,..., WEIGHT_BELOW_CA=1, WEIGHT_C_A=10, WEIGHT_ABOVE_CA=100]
        # # Bewegung muss Kosten verursachen. weight = 0 => ignorieren von motion controller zum ziel_xyz.

        # WILL ONLY BE EVALUATED ONCE, CUZ ITS WRITTEN IN PYTHON
        for i in range(0, len(self.goals_rs)):
            self.minimize_pose('goal_' + str(i), self.get_god_map().to_symbol(
                self.get_identifier() + [self.get_weight_py_f, tuple('goal_' + str(i))]))

        self.add_debug_constraint("debugTime", time)
        self.add_debug_constraint("debugTimeInSecs", time_in_secs)

        self.add_debug_constraint("debugweightATime", self.goals_times[0])
        self.add_debug_constraint("debugweightBTime", self.goals_times[1])

        # self.add_debug_constraint("debugweightA", self.get_weight(self.goal_a, self.goals_times[0], self.goals_times[1]))
        # self.add_debug_constraint("debugActivation_funA", self.activation_fun(self.goals_times[0]))
        self.add_debug_constraint("debugDistanceToA", self.distance_to_goal('goal_0'))

        # self.add_debug_constraint("debugweightB", self.get_weight(self.goal_b, self.goals_times[1], self.goals_times[2]))
        # self.add_debug_constraint("debugActivation_funB", self.activation_fun(self.goals_times[1]))
        self.add_debug_constraint("debugDistanceToB", self.distance_to_goal('goal_1'))

    def activation_fun(self, x):
        # time = self.get_god_map().to_symbol(identifier.time)
        # time_in_secs = self.get_input_sampling_period() * time
        ret = 2 / (1 + w.ca.exp(-x)) - 1
        return w.if_greater(ret, 0, ret, 0)

    def activation_fun_py(self, x):
        ret = 2 / (1 + np.exp(-x)) - 1
        return ret if ret > 0 else 0

    def get_weight(self, p, p_time, p_next_time):
        # p_reached = w.if_greater(p_next_time, p_time, 1, 0)
        # invalid_p_time = w.if_eq(p_time, -1, 1, 0)
        return self.activation_fun(
            p_time - p_next_time)  # w.if_greater_eq(p_reached + invalid_p_time, 1, 0, self.activation_fun(p_time))

    def get_weight_py(self, *p):
        self.update_goal_times()
        p_str = str(p).replace('\'', '').replace(',', '').replace('(', '').replace(')', '').replace(' ', '').replace(
            'u', '')
        for i in range(0, len(self.goals_rs)):
            if 'goal_' + str(i) in p_str:
                if i == len(self.goals_rs) - 1:
                    return self.activation_fun_py(self.goals_times[i])
                else:
                    return self.activation_fun_py(self.goals_times[i] - self.goals_times[i + 1])
        return 0

    def distance_to_goal(self, p):
        return w.norm(w.position_of(self.get_input_PoseStamped(p) - self.get_fk(self.root_link, self.tip_link)))

    def distance_to_goal_py(self, p):
        cur = self.robot.get_fk_pose(self.root_link, self.tip_link).pose.position
        x = p.pose.position.x - cur.x
        y = p.pose.position.y - cur.y
        z = p.pose.position.z - cur.z
        return np.linalg.norm(np.array([x, y, z]))

    def goal_reached(self, p):
        return w.if_less(self.distance_to_goal(p), 0.1, 1, 0)

    def goal_reached_py(self, p):
        return 1 if self.distance_to_goal_py(p) < 0.1 else 0

    # def activation_for_p(self, p, p_time):
    #    return w.if_eq(self.goal_reached(p), 1, 0, self.activation_fun(p_time)) #self.distance_to_last_goal(point, path) #+ self.distance_to_environment(point)

    def distance_to_last_goal(self, p, p_next):
        return WEIGHT_BELOW_CA  # w.if_eq(w.ca.SX.sym(p), w.ca.SX.sym(path[0]), 3, #w.ca.SX.sym(path[0]) breaks stuff, since god_map thinks it has it
        # w.if_eq(w.ca.SX.sym(p), w.ca.SX.sym(path[1]), 2,
        # w.if_eq(w.ca.SX.sym(p), w.ca.SX.sym(path[2]), 1,
        # w.if_eq(w.ca.SX.sym(p), w.ca.SX.sym(path[3]), 0, 0))))

    def distance_to_environment(self, point):
        pass

    def minimize_pose(self, goal, weight):
        self.minimize_position(goal, weight)
        self.minimize_rotation(goal, weight)

    def minimize_position(self, goal, weight):
        r_P_g = w.position_of(self.get_input_PoseStamped(goal))
        max_velocity = self.get_input_float(self.max_linear_velocity)
        max_acceleration = self.get_input_float(self.max_linear_acceleration)

        self.add_minimize_position_constraints(r_P_g, max_velocity, max_acceleration, self.root_link, self.tip_link,
                                               self.goal_constraint, weight, prefix=goal)

    def minimize_rotation(self, goal, weight):
        r_R_g = w.rotation_of(self.get_input_PoseStamped(goal))
        max_velocity = self.get_input_float(self.max_angular_velocity)
        max_acceleration = self.get_input_float(self.max_angular_acceleration)

        self.add_minimize_rotation_constraints(r_R_g, self.root_link, self.tip_link, max_velocity, weight,
                                               self.goal_constraint, prefix=goal)

    def __str__(self):
        s = super(CartesianPathPy, self).__str__()
        return u'{}'.format(s)


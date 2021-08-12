import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w, identifier
from giskardpy.goals.goal import Goal, WEIGHT_COLLISION_AVOIDANCE, WEIGHT_ABOVE_CA


class ExternalCollisionAvoidance(Goal):

    def __init__(self, link_name, max_velocity=0.2, hard_threshold=0.0, soft_threshold=0.05, idx=0,
                 num_repeller=1, **kwargs):
        """
        Don't use me
        """
        self.max_velocity = max_velocity
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold
        self.num_repeller = num_repeller
        self.link_name = link_name
        self.idx = idx
        super(ExternalCollisionAvoidance, self).__init__(**kwargs)
        self.robot_root = self.get_robot().get_root()
        self.robot_name = self.get_robot_unsafe().get_name()

    def get_contact_normal_on_b_in_root(self):
        return self.get_god_map().list_to_vector3(identifier.closest_point + [u'get_external_collisions',
                                                                              (self.link_name,),
                                                                              self.idx,
                                                                         u'get_contact_normal_in_root',
                                                                              tuple()])

    def get_closest_point_on_a_in_a(self):
        return self.get_god_map().list_to_point3(identifier.closest_point + [u'get_external_collisions',
                                                                             (self.link_name,),
                                                                             self.idx,
                                                                        u'get_position_on_a_in_a',
                                                                             tuple()])

    def get_closest_point_on_b_in_root(self):
        return self.get_god_map().list_to_point3(identifier.closest_point + [u'get_external_collisions',
                                                                             (self.link_name,),
                                                                             self.idx,
                                                                        u'get_position_on_b_in_root',
                                                                             tuple()])

    def get_actual_distance(self):
        return self.god_map.to_symbol(identifier.closest_point + [u'get_external_collisions',
                                                                  (self.link_name,),
                                                                  self.idx,
                                                                  u'get_contact_distance',
                                                                  tuple()])

    def get_number_of_external_collisions(self):
        return self.god_map.to_symbol(identifier.closest_point + [u'get_number_of_external_collisions',
                                                                  (self.link_name,)])

    def make_constraints(self):
        a_P_pa = self.get_closest_point_on_a_in_a()
        r_V_n = self.get_contact_normal_on_b_in_root()
        actual_distance = self.get_actual_distance()
        max_velocity = self.get_parameter_as_symbolic_expression(u'max_velocity')
        # zero_weight_distance = self.get_input_float(self.zero_weight_distance)
        hard_threshold = self.get_parameter_as_symbolic_expression(u'hard_threshold')
        soft_threshold = self.get_parameter_as_symbolic_expression(u'soft_threshold')
        # spring_threshold = soft_threshold
        # soft_threshold = soft_threshold * 0.5
        sample_period = self.get_sampling_period_symbol()
        number_of_external_collisions = self.get_number_of_external_collisions()
        num_repeller = self.get_parameter_as_symbolic_expression(u'num_repeller')

        root_T_a = self.get_fk(self.robot_root, self.link_name)

        r_P_pa = w.dot(root_T_a, a_P_pa)

        dist = w.dot(r_V_n.T, r_P_pa)[0]

        qp_limits_for_lba = max_velocity * sample_period * self.control_horizon

        penetration_distance = soft_threshold - actual_distance
        lower_limit = penetration_distance

        lower_limit_limited = w.limit(lower_limit,
                                      -qp_limits_for_lba,
                                      qp_limits_for_lba)

        upper_slack = w.if_greater(actual_distance, hard_threshold,
                                   w.limit(soft_threshold - hard_threshold,
                                           -qp_limits_for_lba,
                                           qp_limits_for_lba),
                                   lower_limit_limited)

        # undo factor in A
        upper_slack /= (sample_period * self.prediction_horizon)
        # upper_slack *= 10

        upper_slack = w.if_greater(actual_distance, 50,  # assuming that distance of unchecked closest points is 100
                                   1e4,
                                   # 1e4,
                                   w.max(0, upper_slack))

        weight = w.if_greater(actual_distance, 50, 0, WEIGHT_COLLISION_AVOIDANCE)

        weight = w.save_division(weight,  # divide by number of active repeller per link
                                 w.min(number_of_external_collisions, num_repeller))

        self.add_constraint(u'/position',
                            reference_velocity=max_velocity,
                            lower_error=lower_limit,
                            upper_error=100,
                            weight=weight,
                            expression=dist,
                            lower_slack_limit=-1e4,
                            upper_slack_limit=upper_slack)

        # self.add_velocity_constraint(u'/position/vel',
        #                              velocity_limit=max_velocity,
        #                              weight=weight,
        #                              expression=dist)

    def __str__(self):
        s = super(ExternalCollisionAvoidance, self).__str__()
        return u'{}/{}/{}'.format(s, self.link_name, self.idx)


class SelfCollisionAvoidance(Goal):

    def __init__(self, link_a, link_b, max_velocity=0.2, hard_threshold=0.0, soft_threshold=0.05, idx=0,
                 num_repeller=1, **kwargs):
        self.link_a = link_a
        self.link_b = link_b
        self.max_velocity = max_velocity
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold
        self.num_repeller = num_repeller
        self.idx = idx
        super(SelfCollisionAvoidance, self).__init__(**kwargs)
        self.robot_root = self.get_robot().get_root()
        self.robot_name = self.get_robot_unsafe().get_name()

    def get_contact_normal_in_b(self):
        return self.get_god_map().list_to_vector3(identifier.closest_point + [u'get_self_collisions',
                                                                              (self.link_a, self.link_b),
                                                                              self.idx,
                                                                         u'get_contact_normal_in_b',
                                                                              tuple()])

    def get_position_on_a_in_a(self):
        return self.get_god_map().list_to_point3(identifier.closest_point + [u'get_self_collisions',
                                                                             (self.link_a, self.link_b),
                                                                             self.idx,
                                                                        u'get_position_on_a_in_a',
                                                                             tuple()])

    def get_b_T_pb(self):
        return self.get_god_map().list_to_translation3(identifier.closest_point + [u'get_self_collisions',
                                                                                   (self.link_a, self.link_b),
                                                                                   self.idx,
                                                                              u'get_position_on_b_in_b',
                                                                                   tuple()])

    def get_actual_distance(self):
        return self.god_map.to_symbol(identifier.closest_point + [u'get_self_collisions',
                                                                  (self.link_a, self.link_b),
                                                                  self.idx,
                                                                  u'get_contact_distance',
                                                                  tuple()])

    def get_number_of_self_collisions(self):
        return self.god_map.to_symbol(identifier.closest_point + [u'get_number_of_self_collisions',
                                                                  (self.link_a, self.link_b)])

    def make_constraints(self):
        max_velocity = self.get_parameter_as_symbolic_expression(u'max_velocity')
        hard_threshold = self.get_parameter_as_symbolic_expression(u'hard_threshold')
        soft_threshold = self.get_parameter_as_symbolic_expression(u'soft_threshold')
        actual_distance = self.get_actual_distance()
        number_of_self_collisions = self.get_number_of_self_collisions()
        num_repeller = self.get_parameter_as_symbolic_expression(u'num_repeller')
        sample_period = self.get_sampling_period_symbol()

        b_T_a = self.get_fk(self.link_b, self.link_a)
        pb_T_b = w.inverse_frame(self.get_b_T_pb())
        a_P_pa = self.get_position_on_a_in_a()

        pb_V_n = self.get_contact_normal_in_b()

        pb_P_pa = w.dot(pb_T_b, b_T_a, a_P_pa)

        dist = w.dot(pb_V_n.T, pb_P_pa)[0]

        weight = w.if_greater(actual_distance, 50, 0, WEIGHT_COLLISION_AVOIDANCE)
        weight = w.save_division(weight,  # divide by number of active repeller per link
                                 w.min(number_of_self_collisions, num_repeller))

        penetration_distance = soft_threshold - actual_distance
        qp_limits_for_lba = max_velocity * sample_period * self.control_horizon

        penetration_distance = soft_threshold - actual_distance
        lower_limit = penetration_distance

        lower_limit_limited = w.limit(lower_limit,
                                      -qp_limits_for_lba,
                                      qp_limits_for_lba)

        upper_slack = w.if_greater(actual_distance, hard_threshold,
                                   w.limit(soft_threshold - hard_threshold,
                                           -qp_limits_for_lba,
                                           qp_limits_for_lba),
                                   lower_limit_limited)

        # undo factor in A
        upper_slack /= (sample_period * self.prediction_horizon)
        # upper_slack *= 10

        upper_slack = w.if_greater(actual_distance, 50,  # assuming that distance of unchecked closest points is 100
                                   1e4,
                                   # 1e4,
                                   w.max(0, upper_slack))

        self.add_constraint(u'/position',
                            reference_velocity=max_velocity,
                            lower_error=lower_limit,
                            upper_error=100,
                            weight=weight,
                            expression=dist,
                            lower_slack_limit=-1e4,
                            upper_slack_limit=upper_slack)

    def __str__(self):
        s = super(SelfCollisionAvoidance, self).__str__()
        return u'{}/{}/{}/{}'.format(s, self.link_a, self.link_b, self.idx)


class CollisionAvoidanceHint(Goal):
    def __init__(self, tip_link, avoidance_hint, object_name, object_link_name, max_linear_velocity=0.1,
                 root_link=None,
                 max_threshold=0.05, spring_threshold=None, weight=WEIGHT_ABOVE_CA, **kwargs):
        """
        This goal pushes the link_name in the direction of avoidance_hint, if it is closer than spring_threshold
        to body_b/link_b.
        :param tip_link: str, name of the robot link, has to have a collision body
        :param avoidance_hint: Vector3Stamped as json, direction in which the robot link will get pushed
        :param object_name: str, name of the environment object, can be the robot, e.g. kitchen
        :param object_link_name: str, name of the link of the environment object. e.g. fridge handle
        :param max_linear_velocity: float, m/s, default 0.1
        :param root_link: str, default robot root, name of the root link for the kinematic chain
        :param max_threshold: float, default 0.05, distance at which the force has reached weight
        :param spring_threshold: float, default max_threshold, need to be >= than max_threshold weight increases from
                                        sprint_threshold to max_threshold linearly, to smooth motions
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        self.link_name = tip_link
        self.key = (tip_link, object_name, object_link_name)
        self.body_b = object_name
        self.link_b = object_link_name
        self.body_b_hash = object_name.__hash__()
        self.link_b_hash = object_link_name.__hash__()
        super(CollisionAvoidanceHint, self).__init__(**kwargs)
        if root_link is None:
            self.root_link = self.get_robot().get_root()
        else:
            self.root_link = root_link

        if spring_threshold is None:
            spring_threshold = max_threshold
        else:
            spring_threshold = max(spring_threshold, max_threshold)

        # register collision checks TODO make function
        added_checks = self.get_god_map().get_data(identifier.added_collision_checks)
        if tip_link in added_checks:
            added_checks[tip_link] = max(added_checks[tip_link], spring_threshold)
        else:
            added_checks[tip_link] = spring_threshold
        self.get_god_map().set_data(identifier.added_collision_checks, added_checks)

        self.avoidance_hint = tf.transform_vector(self.root_link, avoidance_hint)
        self.avoidance_hint.vector = tf.normalize(self.avoidance_hint.vector)

        self.max_velocity = max_linear_velocity
        self.threshold = max_threshold
        self.threshold2 = spring_threshold
        self.weight = weight

    def get_contact_normal_on_b_in_root(self):
        return self.get_god_map().list_to_vector3(identifier.closest_point + [u'get_external_collisions_long_key',
                                                                              self.key,
                                                                         u'get_contact_normal_in_root',
                                                                              tuple()])

    def get_closest_point_on_a_in_a(self):
        return self.get_god_map().list_to_point3(identifier.closest_point + [u'get_external_collisions_long_key',
                                                                             self.key,
                                                                        u'get_position_on_a_in_a',
                                                                             tuple()])

    def get_closest_point_on_b_in_root(self):
        return self.get_god_map().list_to_point3(identifier.closest_point + [u'get_external_collisions_long_key',
                                                                             self.key,
                                                                        u'get_position_on_b_in_root',
                                                                             tuple()])

    def get_actual_distance(self):
        return self.god_map.to_symbol(identifier.closest_point + [u'get_external_collisions_long_key',
                                                                  self.key,
                                                                  u'get_contact_distance',
                                                                  tuple()])

    def get_body_b(self):
        return self.god_map.to_symbol(identifier.closest_point + [u'get_external_collisions_long_key',
                                                                  self.key, u'get_body_b_hash', tuple()])

    def get_link_b(self):
        return self.god_map.to_symbol(identifier.closest_point + [u'get_external_collisions_long_key',
                                                                  self.key, u'get_link_b_hash', tuple()])

    def make_constraints(self):
        weight = self.get_parameter_as_symbolic_expression(u'weight')
        actual_distance = self.get_actual_distance()
        max_velocity = self.get_parameter_as_symbolic_expression(u'max_velocity')
        max_threshold = self.get_parameter_as_symbolic_expression(u'threshold')
        spring_threshold = self.get_parameter_as_symbolic_expression(u'threshold2')
        body_b_hash = self.get_body_b()
        link_b_hash = self.get_link_b()
        actual_distance_capped = w.max(actual_distance, 0)

        root_T_a = self.get_fk(self.root_link, self.link_name)

        spring_error = spring_threshold - actual_distance_capped
        spring_error = w.max(spring_error, 0)

        spring_weight = w.if_eq(spring_threshold, max_threshold, 0,
                                weight * (spring_error / (spring_threshold - max_threshold)) ** 2)

        weight = w.if_less_eq(actual_distance, max_threshold, weight,
                              spring_weight)
        weight = w.if_eq(body_b_hash, self.body_b_hash, weight, 0)
        weight = w.if_eq(link_b_hash, self.link_b_hash, weight, 0)

        root_V_avoidance_hint = self.get_parameter_as_symbolic_expression(u'avoidance_hint')

        # penetration_distance = threshold - actual_distance_capped

        root_P_a = w.position_of(root_T_a)
        expr = w.dot(root_V_avoidance_hint[:3].T, root_P_a[:3])

        # FIXME really?
        self.add_constraint(u'avoidance_hint',
                            reference_velocity=max_velocity,
                            lower_error=max_velocity,
                            upper_error=max_velocity,
                            weight=weight,
                            expression=expr)

    def __str__(self):
        s = super(CollisionAvoidanceHint, self).__str__()
        return u'{}/{}/{}/{}'.format(s, self.link_name, self.body_b, self.link_b)

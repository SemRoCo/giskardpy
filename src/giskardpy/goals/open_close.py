from __future__ import division

from copy import deepcopy

import PyKDL as kdl
import numpy as np
from geometry_msgs.msg import Vector3Stamped

import giskardpy.identifier as identifier
import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w
from giskardpy.goals.cartesian_goals import CartesianPoseStraight
from giskardpy.goals.goal import Goal, WEIGHT_ABOVE_CA
from giskardpy.utils.logging import logwarn


class OpenDoor(Goal):
    def __init__(self, tip_link, object_name, object_link_name, angle_goal, root_link=None,
                 weight=WEIGHT_ABOVE_CA, **kwargs):
        super(OpenDoor, self).__init__(**kwargs)

        self.tip = tip_link
        self.angle_goal = angle_goal
        self.handle_link = object_link_name
        handle_frame_id = u'iai_kitchen/' + object_link_name
        self.object_name = object_name
        if root_link is None:
            self.root = self.get_god_map().get_data(identifier.robot).get_root()
        else:
            self.root = root_link

        environment_object = self.get_world().get_object(object_name)
        self.hinge_joint = environment_object.get_movable_parent_joint(object_link_name)
        hinge_child = environment_object.get_child_link_of_joint(self.hinge_joint)

        hinge_frame_id = u'iai_kitchen/' + hinge_child

        hinge_V_hinge_axis = kdl.Vector(*environment_object.get_joint_axis(self.hinge_joint))
        hinge_V_hinge_axis_msg = Vector3Stamped()
        hinge_V_hinge_axis_msg.header.frame_id = hinge_frame_id
        hinge_V_hinge_axis_msg.vector.x = hinge_V_hinge_axis[0]
        hinge_V_hinge_axis_msg.vector.y = hinge_V_hinge_axis[1]
        hinge_V_hinge_axis_msg.vector.z = hinge_V_hinge_axis[2]

        hingeStart_T_tipStart = tf.msg_to_kdl(tf.lookup_pose(hinge_frame_id, self.tip))

        hinge_pose = tf.lookup_pose(self.root, hinge_frame_id)

        root_T_hingeStart = tf.msg_to_kdl(hinge_pose)
        hinge_T_handle = tf.msg_to_kdl(tf.lookup_pose(hinge_frame_id, handle_frame_id))  # constant
        hinge_joint_current = environment_object.joint_state[self.hinge_joint].position

        hingeStart_P_tipStart = hingeStart_T_tipStart.p

        projection = kdl.dot(hingeStart_P_tipStart, hinge_V_hinge_axis)
        hinge0_P_tipStartProjected = hingeStart_P_tipStart - hinge_V_hinge_axis * projection

        hinge0_T_hingeCurrent = kdl.Frame(kdl.Rotation().Rot(hinge_V_hinge_axis, hinge_joint_current))
        root_T_hinge0 = root_T_hingeStart * hinge0_T_hingeCurrent.Inverse()
        root_T_handleGoal = root_T_hinge0 * kdl.Frame(
            kdl.Rotation().Rot(hinge_V_hinge_axis, angle_goal)) * hinge_T_handle

        handleStart_T_tipStart = tf.msg_to_kdl(tf.lookup_pose(handle_frame_id, self.tip))
        root_T_tipGoal = tf.kdl_to_np(root_T_handleGoal * handleStart_T_tipStart)

        hinge0_T_tipGoal = tf.kdl_to_np(hingeStart_T_tipStart)
        hinge0_T_tipStartProjected = tf.kdl_to_np(kdl.Frame(hinge0_P_tipStartProjected))

        hinge0_P_tipStart_norm = np.linalg.norm(tf.kdl_to_np(hingeStart_P_tipStart))

        self.hinge_pose = hinge_pose
        self.hinge_V_hinge_axis_msg = hinge_V_hinge_axis_msg
        self.hinge0_T_tipGoal = hinge0_T_tipGoal
        self.root_T_hinge0 = tf.kdl_to_np(root_T_hinge0)
        self.hinge0_T_tipStartProjected = hinge0_T_tipStartProjected
        self.root_T_tipGoal = root_T_tipGoal
        self.hinge0_P_tipStart_norm = hinge0_P_tipStart_norm
        self.weight = weight

    def make_constraints(self):
        root_T_tip = self.get_fk(self.root, self.tip)
        root_T_hinge = self.get_parameter_as_symbolic_expression(u'hinge_pose')
        hinge_V_hinge_axis = self.get_parameter_as_symbolic_expression(u'hinge_V_hinge_axis_msg')[:3]
        hinge_T_root = w.inverse_frame(root_T_hinge)
        root_T_tipGoal = self.get_parameter_as_symbolic_expression(u'root_T_tipGoal')
        root_T_hinge0 = self.get_parameter_as_symbolic_expression(u'root_T_hinge0')
        root_T_tipCurrent = self.get_fk_evaluated(self.root, self.tip)
        hinge0_R_tipGoal = w.rotation_of(self.get_parameter_as_symbolic_expression(u'hinge0_T_tipGoal'))
        dist_goal = self.get_parameter_as_symbolic_expression(u'hinge0_P_tipStart_norm')
        hinge0_T_tipStartProjected = self.get_parameter_as_symbolic_expression(u'hinge0_T_tipStartProjected')

        self.add_point_goal_constraints(frame_P_current=w.position_of(root_T_tip),
                                        frame_P_goal=w.position_of(root_T_tipGoal),
                                        reference_velocity=0.1,
                                        weight=self.weight)

        hinge_P_tip = w.position_of(w.dot(hinge_T_root, root_T_tip))[:3]

        dist_expr = w.norm(hinge_P_tip)
        self.add_constraint(name_suffix=u'/dist',
                            reference_velocity=0.1,
                            lower_error=dist_goal - dist_expr,
                            upper_error=dist_goal - dist_expr,
                            weight=self.weight,
                            expression=dist_expr)

        hinge0_T_tipCurrent = w.dot(w.inverse_frame(root_T_hinge0), root_T_tipCurrent)
        hinge0_P_tipStartProjected = w.position_of(hinge0_T_tipStartProjected)
        hinge0_P_tipCurrent = w.position_of(hinge0_T_tipCurrent)[:3]

        projection = w.dot(hinge0_P_tipCurrent.T, hinge_V_hinge_axis)
        hinge0_P_tipCurrentProjected = hinge0_P_tipCurrent - hinge_V_hinge_axis * projection

        current_tip_angle_projected = w.angle_between_vector(hinge0_P_tipStartProjected, hinge0_P_tipCurrentProjected)

        hinge0_T_hingeCurrent = w.rotation_matrix_from_axis_angle(hinge_V_hinge_axis, current_tip_angle_projected)

        root_T_hingeCurrent = w.dot(root_T_hinge0, hinge0_T_hingeCurrent)

        root_R_tipGoal = w.dot(root_T_hingeCurrent, hinge0_R_tipGoal)

        self.add_rotation_goal_constraints(frame_R_current=w.rotation_of(self.get_fk(self.root, self.tip)),
                                           frame_R_goal=root_R_tipGoal,
                                           current_R_frame_eval=w.rotation_of(self.get_fk_evaluated(self.tip, self.root)),
                                           reference_velocity=0.5,
                                           weight=self.weight)

    def __str__(self):
        s = super(OpenDoor, self).__str__()
        return u'{}/{}'.format(s, self.handle_link)


class OpenDrawer(Goal):
    def __init__(self, tip_link, object_name, object_link_name, distance_goal, root_link=None,
                 weight=WEIGHT_ABOVE_CA, **kwargs):
        """
        :type tip_link: str
        :param tip_link: tip of manipulator (gripper) which is used
        :type object_name str
        :param object_name
        :type object_link_name str
        :param object_link_name handle to grasp
        :type distance_goal float
        :param distance_goal
               relative opening distance 0 = close, 1 = fully open
        :type root_link: str
        :param root_link: default is root link of robot
        """
        # Process input parameters
        if root_link is None:
            self.root = self.get_robot().get_root()
        else:
            self.root = root_link
        self.tip = tip_link

        self.distance_goal = distance_goal

        self.handle_link = object_link_name
        handle_frame_id = u'iai_kitchen/' + object_link_name

        self.object_name = object_name
        super(OpenDrawer, self).__init__(**kwargs)

        environment_object = self.get_world().get_object(object_name)
        # Get movable joint
        self.hinge_joint = environment_object.get_movable_parent_joint(object_link_name)
        # Child of joint
        hinge_child = environment_object.get_child_link_of_joint(self.hinge_joint)

        hinge_frame_id = u'iai_kitchen/' + hinge_child

        # Get movable axis of drawer (= prismatic joint)
        hinge_drawer_axis = kdl.Vector(*environment_object.get_joint_axis(self.hinge_joint))
        hinge_drawer_axis_msg = Vector3Stamped()
        hinge_drawer_axis_msg.header.frame_id = hinge_frame_id
        hinge_drawer_axis_msg.vector.x = hinge_drawer_axis[0]
        hinge_drawer_axis_msg.vector.y = hinge_drawer_axis[1]
        hinge_drawer_axis_msg.vector.z = hinge_drawer_axis[2]

        # Get joint limits TODO: check if desired goal is within limits
        min_limit, max_limit = environment_object.get_joint_position_limits(
            self.hinge_joint)
        current_joint_pos = environment_object.joint_state[self.hinge_joint].position

        # Avoid invalid values
        if distance_goal < min_limit:
            self.distance_goal = min_limit
        if distance_goal > max_limit:
            self.distance_goal = max_limit

        hinge_frame_id = u'iai_kitchen/' + hinge_child

        # Get frame of current tip pose
        root_T_tip_current = tf.msg_to_kdl(tf.lookup_pose(self.root, tip_link))
        hinge_drawer_axis_kdl = tf.msg_to_kdl(hinge_drawer_axis_msg)  # get axis of joint
        # Get transform from hinge to root
        root_T_hinge = tf.msg_to_kdl(tf.lookup_pose(self.root, hinge_frame_id))

        # Get translation vector from current to goal position
        tip_current_V_tip_goal = hinge_drawer_axis_kdl * (self.distance_goal - current_joint_pos)

        root_V_hinge_drawer = root_T_hinge.M * tip_current_V_tip_goal  # get vector in hinge frame
        root_T_tip_goal = deepcopy(root_T_tip_current)  # copy object to manipulate it
        # Add translation vector to current position (= get frame of goal position)
        root_T_tip_goal.p += root_V_hinge_drawer

        # Convert goal pose to dict for Giskard
        root_T_tip_goal_dict = tf.kdl_to_pose_stamped(root_T_tip_goal, self.root)

        self.add_constraints_of_goal(CartesianPoseStraight(self.root,
                                                           self.tip,
                                                           root_T_tip_goal_dict,
                                                           weight=weight))

    def __str__(self):
        s = super(OpenDrawer, self).__str__()
        return u'{}/{}/{}'.format(s, self.root, self.tip)


class Open(Goal):
    def __init__(self, tip_link, object_name, object_link_name, root_link=None, goal_joint_state=None,
                 weight=WEIGHT_ABOVE_CA, **kwargs):
        super(Open, self).__init__(**kwargs)
        environment_object = self.get_world().get_object(object_name)
        joint_name = environment_object.get_movable_parent_joint(object_link_name)

        if environment_object.is_joint_revolute(joint_name) or environment_object.is_joint_prismatic(joint_name):
            min_limit, max_limit = environment_object.get_joint_position_limits(joint_name)
            if goal_joint_state:
                goal_joint_state = min(max_limit, goal_joint_state)
            else:
                goal_joint_state = max_limit

        if environment_object.is_joint_revolute(joint_name):
            self.add_constraints_of_goal(OpenDoor(tip_link=tip_link,
                                                  object_name=object_name,
                                                  object_link_name=object_link_name,
                                                  angle_goal=goal_joint_state,
                                                  root_link=root_link,
                                                  weight=weight, **kwargs))
        elif environment_object.is_joint_prismatic(joint_name):
            self.add_constraints_of_goal(OpenDrawer(tip_link=tip_link,
                                                    object_name=object_name,
                                                    object_link_name=object_link_name,
                                                    distance_goal=goal_joint_state,
                                                    root_link=root_link,
                                                    weight=weight, **kwargs))
        else:
            logwarn(u'Opening containers with joint of type "{}" not supported'.format(
                environment_object.get_joint_type(joint_name)))


class Close(Goal):
    def __init__(self, tip_link, object_name, object_link_name, root_link=None, goal_joint_state=None,
                 weight=WEIGHT_ABOVE_CA, **kwargs):
        super(Close, self).__init__(**kwargs)
        environment_object = self.get_world().get_object(object_name)
        joint_name = environment_object.get_movable_parent_joint(object_link_name)

        if environment_object.is_joint_revolute(joint_name) or environment_object.is_joint_prismatic(joint_name):
            min_limit, max_limit = environment_object.get_joint_position_limits(joint_name)
            if goal_joint_state:
                goal_joint_state = max(min_limit, goal_joint_state)
            else:
                goal_joint_state = min_limit

        if environment_object.is_joint_revolute(joint_name):
            self.add_constraints_of_goal(OpenDoor(tip_link=tip_link,
                                                  object_name=object_name,
                                                  object_link_name=object_link_name,
                                                  angle_goal=goal_joint_state,
                                                  root_link=root_link,
                                                  weight=weight, **kwargs))
        elif environment_object.is_joint_prismatic(joint_name):
            self.add_constraints_of_goal(OpenDrawer(tip_link=tip_link,
                                                    object_name=object_name,
                                                    object_link_name=object_link_name,
                                                    distance_goal=goal_joint_state,
                                                    root_link=root_link,
                                                    weight=weight, **kwargs))
        else:
            logwarn(u'Opening containers with joint of type "{}" not supported'.format(
                environment_object.get_joint_type(joint_name)))

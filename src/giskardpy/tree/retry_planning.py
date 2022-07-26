import json
from copy import deepcopy

import actionlib
import numpy as np
import rospy
import yaml
from py_trees import Status

import giskardpy.identifier as identifier
import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import Constraint, MoveGoal, MoveAction
from giskard_msgs.srv import GlobalPathNeededRequest, GlobalPathNeeded
from giskardpy.exceptions import PlanningException
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.utils import convert_dictionary_to_ros_message, msg_to_list


class RetryPlanning(GiskardBehavior):

    def __init__(self, name):
        super(RetryPlanning, self).__init__(name)
        self.path_constraint_name = 'CartesianPathCarrot'
        self.valid = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

    @profile
    def update(self):
        e = self.get_blackboard_exception()
        if e and self.must_replan(e):
            self.clear_blackboard_exception()
            self.get_god_map().set_data(identifier.global_planner_needed, True)
            return Status.RUNNING
        elif self.is_reaching_goal_pose_trivial():
            logging.loginfo(f'{self.path_constraint_name} terminated early, but goal can be reached trivially.')
            logging.loginfo(u'Solving rest with CartesianPose.')
            self.send_trivial_cartesian_pose_goal()
            return Status.RUNNING
        elif self.cartesian_path_planning_failed():
            self.must_replan(PlanningException())
            logging.loginfo(f'{self.path_constraint_name} did not traverse the planned path.')
            logging.loginfo(u'Replanning a new path.')
            self.get_god_map().set_data(identifier.global_planner_needed, True)
            return Status.RUNNING
        else:
            return Status.SUCCESS

    def send_trivial_cartesian_pose_goal(self):
        c, nc = self.move_parameter_value_pair_to_constraint(self.path_constraint_name, 'CartesianPose',
                                                             parameters=['root_link', 'tip_link', 'goal'])
        move_cmd = self.god_map.get_data(identifier.next_move_goal)  # type: MoveCmd
        m = MoveGoal()
        move_cmd.constraints.remove(c)
        move_cmd.constraints.append(nc)
        m.cmd_seq.append(move_cmd)
        m.type = MoveGoal.PLAN_AND_EXECUTE
        client = actionlib.SimpleActionClient("/giskard/command", MoveAction)
        client.send_goal(m)

    def move_parameter_value_pair_to_constraint(self, from_type, to_type, parameters=None):

        move_cmd = self.god_map.get_data(identifier.next_move_goal)  # type: MoveCmd
        if any([c.type in [from_type] for c in move_cmd.constraints]):
            for c in move_cmd.constraints:
                if c.type == from_type:
                    npvps_d = dict()
                    pvps = yaml.load(c.parameter_value_pair)
                    for k in parameters:
                        npvps_d[k] = deepcopy(pvps[k])
                    nc = Constraint()
                    nc.type = to_type
                    nc.parameter_value_pair = json.dumps(npvps_d)
                    return c, nc
        else:
            raise KeyError('Could not find constraint of type {} in move_cmd.'.format(from_type))

    def is_reaching_goal_pose_trivial(self):
        move_cmd = self.god_map.get_data(identifier.next_move_goal)  # type: MoveCmd
        if any([c.type in [self.path_constraint_name] for c in move_cmd.constraints]):
            global_move_cmd = deepcopy(move_cmd)
            global_move_cmd.constraints = list()
            for c in move_cmd.constraints:
                if c.type == self.path_constraint_name:
                    d = yaml.load(c.parameter_value_pair)
                    if 'goals' in d:
                        goal_pose = convert_dictionary_to_ros_message(d['goal']).pose
                        rospy.wait_for_service('~is_global_path_needed', timeout=5.0)
                        is_global_path_needed = rospy.ServiceProxy('~is_global_path_needed', GlobalPathNeeded)
                        req = GlobalPathNeededRequest()
                        req.root_link = d['root_link']
                        req.tip_link = d['tip_link']
                        req.env_group = 'kitchen'
                        req.pose_goal = goal_pose
                        req.simple = True
                        return not is_global_path_needed(req).needed
                    else:
                        return False

    def cartesian_path_planning_failed(self):
        # TODO: check if goal in c is eq to c.goals[-1]
        # Check if the robot reached the intended goal
        ret = False
        move_cmd = self.god_map.get_data(identifier.next_move_goal)  # type: MoveCmd
        if any([c.type in [self.path_constraint_name] for c in move_cmd.constraints]):
            global_move_cmd = deepcopy(move_cmd)
            global_move_cmd.constraints = list()
            for c in move_cmd.constraints:
                if c.type == self.path_constraint_name:
                    d = yaml.load(c.parameter_value_pair)
                    if 'goals' in d:
                        goal_pose = convert_dictionary_to_ros_message(d['goals'][-1]).pose
                        calculated_goal = tf.homo_matrix_to_pose(self.world.get_fk('map', d['tip_link']))
                        goal_pose_arr = np.array(msg_to_list(goal_pose))
                        calculated_goal_arr = np.array(msg_to_list(calculated_goal))
                        res = abs(goal_pose_arr - calculated_goal_arr)
                        ret |= (res > self.valid).any()
                    else:
                        return True
        return ret

    def must_replan(self, exception):
        """
        :type exception: Exception
        :rtype: int
        """

        if isinstance(exception, PlanningException):
            #supported_global_cart_goals = ['CartesianPose', 'CartesianPosition', 'CartesianPreGrasp']
            failed_move_cmd = self.god_map.get_data(identifier.next_move_goal) # type: MoveCmd
            global_move_cmd = deepcopy(failed_move_cmd)
            global_move_cmd.constraints = list()
            for c in failed_move_cmd.constraints:
                #if c.type in supported_global_cart_goals:
                #    logging.loginfo(u'Replanning a new path for CartesianPose.')
                #    n_c = Constraint()
                #    n_c.type = 'CartesianPose'
                #    n_c.parameter_value_pair = c.parameter_value_pair
                #    global_move_cmd.constraints.append(n_c)
                if c.type == self.path_constraint_name:
                    logging.loginfo(f'Replanning a new path for {self.path_constraint_name}.')
                    n_c = Constraint()
                    n_c.type = self.path_constraint_name
                    d = yaml.load(c.parameter_value_pair)
                    if 'goals' in d:
                        d.pop('goals')
                    n_c.parameter_value_pair = yaml.dump(d)
                    global_move_cmd.constraints.append(n_c)
                    self.get_god_map().set_data(identifier.global_planner_needed, True)
                elif c.type == 'CartesianPreGrasp':
                    logging.loginfo(u'Resampling a new PreGrasp pose.')
                    n_c = Constraint()
                    n_c.type = 'CartesianPreGrasp'
                    d = yaml.load(c.parameter_value_pair)
                    d.pop('goal')
                    n_c.parameter_value_pair = yaml.dump(d)
                    global_move_cmd.constraints.append(n_c)
                else:
                    global_move_cmd.constraints.append(c)
            self.get_god_map().set_data(identifier.next_move_goal, global_move_cmd)
            return True
        else:
            return False

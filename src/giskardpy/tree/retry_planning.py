from copy import deepcopy

import numpy as np
import rospy
import yaml
from py_trees import Status

import giskardpy.identifier as identifier
import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import Constraint
from giskardpy.exceptions import PlanningException
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.utils import convert_dictionary_to_ros_message, msg_to_list


class RetryPlanning(GiskardBehavior):

    def __init__(self, name):
        super(RetryPlanning, self).__init__(name)
        self.valid = np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])

    @profile
    def update(self):

        e = self.get_blackboard_exception()
        if e and self.must_replan(e):
            self.clear_blackboard_exception()
            logging.loginfo(u'Replanning with global planner.')
            self.get_god_map().set_data(identifier.global_planner_needed, True)
            return Status.RUNNING
        elif self.cartesian_path_planning_failed():
            self.must_replan(PlanningException())
            logging.loginfo(u'CartesianPath did not follow the the path to the goal.')
            logging.loginfo(u'Replanning a new path with global planner.')
            self.get_god_map().set_data(identifier.global_planner_needed, True)
            return Status.RUNNING
        else:
            return Status.SUCCESS

    def cartesian_path_planning_failed(self):
        # TODO: check if goal in c is eq to c.goals[-1]
        # Check if the robot reached the intended goal
        ret = False
        move_cmd = self.god_map.get_data(identifier.next_move_goal)  # type: MoveCmd
        if any([c.type in ['CartesianPathCarrot'] for c in move_cmd.constraints]):
            global_move_cmd = deepcopy(move_cmd)
            global_move_cmd.constraints = list()
            for c in move_cmd.constraints:
                if c.type == 'CartesianPathCarrot':
                    d = yaml.load(c.parameter_value_pair)
                    goal_pose = convert_dictionary_to_ros_message(d['goals'][-1]).pose
                    calculated_goal = tf.homo_matrix_to_pose(self.world.get_fk('map', d['tip_link']))
                    goal_pose_arr = np.array(msg_to_list(goal_pose))
                    calculated_goal_arr = np.array(msg_to_list(calculated_goal))
                    res = abs(goal_pose_arr - calculated_goal_arr)
                    ret |= (res > self.valid).any()
        return ret

    def must_replan(self, exception):
        """
        :type exception: Exception
        :rtype: int
        """

        if isinstance(exception, PlanningException):
            supported_global_cart_goals = ['CartesianPose', 'CartesianPosition']
            failed_move_cmd = self.god_map.get_data(identifier.next_move_goal) # type: MoveCmd
            if any([c.type in supported_global_cart_goals for c in failed_move_cmd.constraints]):
                global_move_cmd = deepcopy(failed_move_cmd)
                global_move_cmd.constraints = list()
                for c in failed_move_cmd.constraints:
                    if c.type in supported_global_cart_goals:
                        n_c = Constraint()
                        n_c.type = 'CartesianPose'
                        n_c.parameter_value_pair = c.parameter_value_pair
                        global_move_cmd.constraints.append(n_c)
                    elif c.type == 'CartesianPathCarrot':
                        n_c = Constraint()
                        n_c.type = 'CartesianPathCarrot'
                        d = yaml.load(c.parameter_value_pair)
                        goal = d['goals'][-1]
                        d.pop('goals')
                        d['goal'] = goal
                        n_c.parameter_value_pair = yaml.dump(d)
                        global_move_cmd.constraints.append(n_c)
                    else:
                        global_move_cmd.constraints.append(c)
                self.get_god_map().set_data(identifier.next_move_goal, global_move_cmd)
                self.get_god_map().set_data(identifier.global_planner_needed, True)
            return True
        else:
            return False

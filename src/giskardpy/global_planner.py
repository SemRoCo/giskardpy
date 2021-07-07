#!/usr/bin/env python


# Author: Mark Moll
from py_trees import Status
from rospy_message_converter.message_converter import convert_dictionary_to_ros_message

import giskardpy.identifier as identifier
from giskardpy.plugin_action_server import GetGoal
from giskardpy.tfwrapper import lookup_transform, get_full_frame_name

from mpl_toolkits.mplot3d import Axes3D
import numpy
import matplotlib.pyplot as plt
from ompl import base as ob
from ompl import geometric as og


class GlobalPlanner(GetGoal):

    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)

        self.kitchen_space = self.create_kitchen_space()
        self.kitchen_floor_space = self.create_kitchen_floor_space()

        self.translation_goal = None
        #self.rotation_goal = None

    def update(self):

        move_cmd = self.get_god_map().get_data(identifier.next_move_goal)  # type: MoveCmd
        if not move_cmd:
            return Status.FAILURE

        cart_cmds = move_cmd.cmd_seq.CartesianConstraint
        if not cart_cmds:
            return Status.FAILURE

        try:
            self.translation_goal = next(cart_c for cart_c in cart_cmds if u'CartesianPosition' in cart_c.type)
            #self.rotation_goal = next(cart_c for cart_c in cart_cmds if u'CartesianOrientationSlerp' in cart_c.type)
        except StopIteration:
            return Status.FAILURE

        if self.translation_goal.parameter_value_pair[u'root_link'] == self.get_robot().get_root():
            trajectory = self.planNavigation()
            if not trajectory:
                return Status.FAILURE
            pass
        else:
            pass #return self.planMovement()

        return Status.SUCCESS

    def isStateValid(self, state):
        # Some arbitrary condition on the state (note that thanks to
        # dynamic type checking we can just call getX() and do not need
        # to convert state to an SE2State.)
        return True # state.getX() < .6

    def create_kitchen_space(self):
        # create an SE3 state space
        space = ob.SE3StateSpace()

        # set lower and upper bounds
        bounds = ob.RealVectorBounds(3)
        bounds.setLow(-1)
        bounds.setHigh(1)
        space.setBounds(bounds)

        return space

    def create_kitchen_floor_space(self):
        # create an SE2 state space
        space = ob.SE2StateSpace()

        # set lower and upper bounds
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(-15)
        bounds.setHigh(15)
        space.setBounds(bounds)

        return space

    def get_robot_global_state(self):
        state = ob.State(self.kitchen_floor_space)
        tf = lookup_transform(self.get_robot().get_root(), identifier.map_frame)
        state().setX(tf.transform.translation.x)
        state().setY(tf.transform.translation.y)
        return state

    def get_translation_goal_state(self, space):
        state = ob.State(space)
        goal_dict = self.translation_goal.parameter_value_pair[u'goal']
        goal =  convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped', goal_dict)
        state().setX(goal.pose.translation.x)
        state().setY(goal.pose.translation.y)
        if is_3D(space):
            state().setZ(goal.pose.translation.Z)
        return state

    def planNavigation(self, plot=True):

        # create a simple setup object
        ss = og.SimpleSetup(self.kitchen_floor_space)
        ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))

        start = self.get_robot_global_state()
        goal = self.get_translation_goal_state(self.kitchen_floor_space)

        ss.setStartAndGoalStates(start, goal)

        # this will automatically choose a default planner with
        # default parameters
        solved = ss.solve(1.0)

        if solved:
            # try to shorten the path
            ss.simplifySolution()
            # print the simplified path
            data = states_matrix_str2array_floats(ss.getSolutionPath().printAsMatrix()) # [[x, y, theta]]
            if plot:
                fig = plt.figure()
                ax = fig.gca(projection='2d')
                ax.plot(data[:, 0], data[:, 1], '.-')
                plt.show()
            return data
        return None

def is_3D(space):
    return type(space) == type(ob.SE3StateSpace())

def states_matrix_str2array_floats(str: str, line_sep='\n', float_sep=' '):
    states_strings = str.split(line_sep)
    while '' in states_strings:
        states_strings.remove('')
    return numpy.array(list(map(lambda x: numpy.fromstring(x, dtype=float, sep=float_sep), states_strings)))

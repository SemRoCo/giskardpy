#!/usr/bin/env python


# Author: Mark Moll
import json
import threading

import rospy
import yaml
from geometry_msgs.msg import PoseStamped, Quaternion
from giskard_msgs.msg import Constraint
from nav_msgs.srv import GetMap
from py_trees import Status
import pybullet as p

from copy import deepcopy
from tf.transformations import quaternion_about_axis

import giskardpy.identifier as identifier
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.tree.plugin_action_server import GetGoal
from giskardpy.utils.tfwrapper import transform_pose

from mpl_toolkits.mplot3d import Axes3D
import numpy
import matplotlib.pyplot as plt
from ompl import base as ob
from ompl import geometric as og

from giskardpy.data_types import JointStates
from giskardpy.utils.utils import convert_dictionary_to_ros_message, convert_ros_message_to_dictionary, \
    replace_jsons_with_ros_messages


class TwoDimRayMotionValidator(ob.MotionValidator):

    def __init__(self, si):
        ob.MotionValidator.__init__(self, si)
        self.lock = threading.Lock()
        self.state_validator = TwoDimStateValidator(si)

    def checkMotion(self, s1, s2):
        with self.lock:
            x_b = s1.getX()
            y_b = s1.getY()
            x_e = s2.getX()
            y_e = s2.getY()
            # Shoot ray from start to end pose and check if it intersects with the kitchen,
            # if so return false, else true.
            d, u = ((-0.33712110805511475, -0.33709930224943446, 0),
                    (0.33712110805511475, 0.33714291386079503, 0))#p.getAABB(self.pybullet_robot_id, 3)
            d_l = (d[0], u[1], 0)
            u_r = (u[0], d[1], 0)
            aabbs = (d_l, d, u, u_r)
            query_b = [[x_b+aabb[0], y_b+aabb[1], 0] for aabb in aabbs]
            query_e = [[x_e+aabb[0], y_e+aabb[1], 0] for aabb in aabbs]
            query_res = p.rayTestBatch(query_b, query_e)
            return all([obj != self.state_validator.pybullet_kitchen_id for obj, _, _, _, _ in query_res]) and \
                   self.state_validator.isValid(s1) and self.state_validator.isValid(s2)


class TwoDimStateValidator(ob.StateValidityChecker, GiskardBehavior):

    def __init__(self, si):
        ob.StateValidityChecker.__init__(self, si)
        GiskardBehavior.__init__(self, str(self))
        self.lock = threading.Lock()
        self.init_map()
        self.init_pybullet_ids_and_joints()

    def init_pybullet_ids_and_joints(self):
        self.pybullet_joints = []
        self.pybullet_robot_id = 2
        self.pybullet_kitchen_id = 3
        for i in range(0, p.getNumJoints(self.get_robot().get_pybullet_id())):
            j = p.getJointInfo(self.get_robot().get_pybullet_id(), i)
            if j[2] != p.JOINT_FIXED:
                self.pybullet_joints.append(j[1].decode('UTF-8'))
                # self.rotation_goal = None

    def init_map(self):
        rospy.wait_for_service('static_map')
        try:
            get_map = rospy.ServiceProxy('static_map', GetMap)
            map = get_map().map
            info = map.info
            tmp = numpy.zeros((info.height, info.width))
            for x_i in range(0, info.height):
                for y_i in range(0, info.width):
                    tmp[x_i][y_i] = map.data[y_i + x_i * info.width]
            self.occ_map = numpy.fliplr(deepcopy(tmp))
            self.occ_map_res = info.resolution
            self.occ_map_origin = info.origin.position
            self.occ_map_height = info.height
            self.occ_map_width = info.width
        except rospy.ServiceException as e:
            rospy.logerr("Failed to get static occupancy map.")

    def is_driveable(self, state):
        x = numpy.sqrt((state.getX() - self.occ_map_origin.x) ** 2)
        y = numpy.sqrt((state.getY() - self.occ_map_origin.y) ** 2)
        if int(y / self.occ_map_res) >= self.occ_map.shape[0] or \
                self.occ_map_width - int(x / self.occ_map_res) >= self.occ_map.shape[1]:
            return False
        return 0 <= self.occ_map[int(y / self.occ_map_res)][self.occ_map_width - int(x / self.occ_map_res)] < 1

    def isValid(self, state):
        # Some arbitrary condition on the state (note that thanks to
        # dynamic type checking we can just call getX() and do not need
        # to convert state to an SE2State.)
        with self.lock:
            x = state.getX()
            y = state.getY()
            # Get current joint states from Bullet and copy them
            current_js = self.get_god_map().get_data(identifier.joint_states)
            new_js = deepcopy(current_js)
            robot = self.get_robot()
            # Calc IK for navigating to given state and ...
            poses = [[x, y, 0], [x + 0.1, y, 0], [x - 0.1, y, 0], [x, y + 0.1, 0], [x, y - 0.1, 0]]
            state_iks = list(map(lambda pose :
                                 p.calculateInverseKinematics(robot.get_pybullet_id(),
                                                              robot.get_pybullet_link_id(u'base_footprint'),
                                                              pose), poses))
            # override on current joint states.
            collisions = ()
            for state_ik in state_iks:
                new_js = JointStates()
                for j_name, j_state in zip(self.pybullet_joints, state_ik):
                    new_js[j_name].position = j_state
                # Set new joint states in Bullet
                self.get_god_map().set_data(identifier.joint_states, new_js)
                # Check if kitchen is colliding with robot
                p.performCollisionDetection()
                collisions += p.getContactPoints(self.pybullet_robot_id, self.pybullet_kitchen_id)
                # Reset joint state
                self.get_god_map().set_data(identifier.joint_states, current_js)
            return collisions is () and self.is_driveable(state)

    def __str__(self):
        return u'TwoDimStateValidator'

class GlobalPlanner(GetGoal):

    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)

        self.kitchen_space = self.create_kitchen_space()
        self.kitchen_floor_space = self.create_kitchen_floor_space()

        self.pose_goal = None
        self.goal_dict = None

    def get_cart_goal(self, cmd):
        try:
            return next(c for c in cmd.constraints if c.type == "CartesianPose")
        except StopIteration:
            return None

    def update(self):

        move_cmd = self.get_god_map().get_data(identifier.next_move_goal)  # type: MoveCmd
        if not move_cmd:
            return Status.FAILURE

        cart_c = self.get_cart_goal(move_cmd)
        if not cart_c:
            return Status.SUCCESS

        self.goal_dict = yaml.load(cart_c.parameter_value_pair)
        ros_pose = convert_dictionary_to_ros_message(self.goal_dict[u'goal'])
        self.pose_goal = transform_pose(self.get_god_map().get_data(identifier.map_frame), ros_pose)

        if self.goal_dict[u'tip_link'] == u'base_footprint':
            trajectory = self.planNavigation()
            if trajectory is None or not trajectory.any():
                return Status.FAILURE
            poses = []
            for i, point in enumerate(trajectory):
                if i == 0:
                    continue # we do not to reach the first pose, since it is the start pose
                base_pose = PoseStamped()
                base_pose.header.frame_id = self.get_god_map().get_data(identifier.map_frame)
                base_pose.pose.position.x = point[0]
                base_pose.pose.position.y = point[1]
                base_pose.pose.position.z = 0
                base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
                poses.append(base_pose)
            c = self.get_cartesian_path_constraint(poses)
            move_cmd.constraints = []
            move_cmd.constraints.append(c)
            self.get_god_map().set_data(identifier.next_move_goal, move_cmd)
            pass
        else:
            pass  # return self.planMovement()

        return Status.SUCCESS

    def get_cartesian_path_constraint(self, poses):
        d = {}
        d[u'parameter_value_pair'] = deepcopy(self.goal_dict)
        d[u'parameter_value_pair'].pop(u'goal')
        d[u'parameter_value_pair'][u'goals'] = list(map(convert_ros_message_to_dictionary, poses))
        # d[u'parameter_value_pair'] = json.dumps(d[u'parameter_value_pair'])
        # d[u'parameter_value_pair'] = json.loads(d[u'parameter_value_pair'])
        c = Constraint()
        c.type = u'CartesianPathCarrot'
        c.parameter_value_pair = json.dumps(d[u'parameter_value_pair'])
        return c

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
        # space.setLongestValidSegmentFraction(0.02)

        # set lower and upper bounds
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(-15)
        bounds.setHigh(15)
        space.setBounds(bounds)

        return space

    def get_translation_start_state(self, space):
        state = ob.State(space)
        pose_root_linkTtip_link = self.get_robot().get_fk_pose(self.goal_dict[u'root_link'],
                                                               self.goal_dict[u'tip_link'])
        pose_mTtip_link = transform_pose(self.get_god_map().get_data(identifier.map_frame), pose_root_linkTtip_link)
        state().setX(pose_mTtip_link.pose.position.x)
        state().setY(pose_mTtip_link.pose.position.y)
        if is_3D(space):
            state().setZ(pose_mTtip_link.pose.position.Z)
        return state

    def get_translation_goal_state(self, space):
        state = ob.State(space)
        state().setX(self.pose_goal.pose.position.x)
        state().setY(self.pose_goal.pose.position.y)
        if is_3D(space):
            state().setZ(self.pose_goal.pose.position.Z)
        return state

    def allocOBValidStateSampler(si):
        return ob.GaussianValidStateSampler(si)

    def getPlanner(self, si):
        # Navigation:
        # RRTstar, RRTsharp: very sparse(= path length eq max range), high number of fp, slow relative to RRTConnect
        # RRTConnect: many points, low number of fp
        # PRMstar: no set_range function, finds no solutions
        # ABITstar: slow, sparse, but good
        planner = og.RRTConnect(si)
        planner.setRange(0.1)
        # planner.setIntermediateStates(True)
        # planner.setup()
        return planner

    def planNavigation(self, num_of_tries=3, plot=True):

        # create a simple setup object
        ss = og.SimpleSetup(self.kitchen_floor_space)

        start = self.get_translation_start_state(self.kitchen_floor_space)
        goal = self.get_translation_goal_state(self.kitchen_floor_space)

        ss.setStartAndGoalStates(start, goal)

        si = ss.getSpaceInformation()
        si.setMotionValidator(TwoDimRayMotionValidator(si))
        si.setStateValidityChecker(TwoDimStateValidator(si))
        si.setup()

        # si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(self.allocOBValidStateSampler))
        # si.setStateValidityCheckingResolution(0.001)

        # this will automatically choose a default planner with
        # default parameters
        planner = self.getPlanner(si)
        ss.setPlanner(planner)

        solved = ob.PlannerStatus.UNKNOWN
        i = 0
        while i < num_of_tries and solved not in [ob.PlannerStatus.APPROXIMATE_SOLUTION, ob.PlannerStatus.EXACT_SOLUTION]:
            solved = ss.solve(20.0)
            i += 1
        #og.PathSimplifier(si).smoothBSpline(ss.getSolutionPath()) # takes around 20-30 secs with RTTConnect(0.1)
        #og.PathSimplifier(si).reduceVertices(ss.getSolutionPath())
        ss.getSolutionPath().interpolate(20)
        if solved:
            # try to shorten the path
            # ss.simplifySolution() DONT! NO! DONT UNCOMMENT THAT! NO! STOP IT! FIRST IMPLEMENT CHECKMOTION! THEN TRY AGAIN!
            # print the simplified path
            data = states_matrix_str2array_floats(ss.getSolutionPath().printAsMatrix())  # [[x, y, theta]]
            if plot:
                plt.close()
                fig, ax = plt.subplots()
                ax.plot(data[:, 1], data[:, 0])
                ax.invert_xaxis()
                ax.set(xlabel='y (m)', ylabel='x (m)',
                       title='Navigation Path in map')
                # ax = fig.gca(projection='2d')
                # ax.plot(data[:, 0], data[:, 1], '.-')
                plt.show()
            return data
        return None


# class MyMotion(ob.MotionValidator):
#
#    def checkMotion(self, s1, s2):
#        return isMotionValidForNavigation_s(s1, s2)


def isStateValidForNavigation_s(state):
    x = state.getX()
    y = state.getY()
    return not (-0.5 > x > -1.5 and 1 < y < 100)


def isMotionValidForNavigation_s(state_a, state_b):
    return isStateValidForNavigation_s(state_a) and isStateValidForNavigation_s(state_b)


def is_3D(space):
    return type(space) == type(ob.SE3StateSpace())


def states_matrix_str2array_floats(str: str, line_sep='\n', float_sep=' '):
    states_strings = str.split(line_sep)
    while '' in states_strings:
        states_strings.remove('')
    return numpy.array(list(map(lambda x: numpy.fromstring(x, dtype=float, sep=float_sep), states_strings)))

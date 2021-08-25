#!/usr/bin/env python

import json
import sys
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

from giskardpy.utils.utils import convert_dictionary_to_ros_message, convert_ros_message_to_dictionary


class PathLengthAndGoalOptimizationObjective(ob.PathLengthOptimizationObjective):

    def __init__(self, si, goal):
        ob.PathLengthOptimizationObjective.__init__(self, si)
        self.goal = goal()

    def stateCost(self, state):
        return ob.Cost(self.getSpaceInformation().distance(state, self.goal))

    def motionCost(self, s1, s2):
        return self.motionCostHeuristic(s1, s2)

    def motionCostHeuristic(self, s1, s2):
        return ob.Cost(self.getSpaceInformation().distance(s1, s2))


class TwoDimRayMotionValidator(ob.MotionValidator, GiskardBehavior):

    def __init__(self, si, object_in_motion):
        ob.MotionValidator.__init__(self, si)
        GiskardBehavior.__init__(self, str(self))
        if 'pybullet' not in sys.modules:
            raise Exception('For two dimensional motion validation the python module pybullet is needed.')
        self.lock = threading.Lock()
        self.object_in_motion = object_in_motion
        self.init_ignore_objects()

    def init_ignore_objects(self, ignore_objects=None):
        if ignore_objects is None:
            ignore_objects = self.get_world().hidden_objects
        else:
            ignore_objects.extend(self.get_world().hidden_objects)
        self.ignore_object_ids = [self.object_in_motion.get_pybullet_id()]
        for ignore_object in ignore_objects:
            if ignore_object in self.get_world()._objects:
                obj = self.get_world()._objects[ignore_object]
                self.ignore_object_ids.append(obj.get_pybullet_id())
            else:
                raise KeyError('Cannot find object of name {} in the PyBulletWorld.'.format(ignore_object))

    def checkMotion(self, s1, s2):
        # checkMotion calls checkMotion with Map anc checkMotion with Bullet
        with self.lock:
            x_b = s1.getX()
            y_b = s1.getY()
            x_e = s2.getX()
            y_e = s2.getY()
            # Shoot ray from start to end pose and check if it intersects with the kitchen,
            # if so return false, else true.
            d, u = ((-0.33712110805511475, -0.33709930224943446, 0),
                    (0.33712110805511475, 0.33714291386079503, 0))  # p.getAABB(self.pybullet_robot_id, 3)
            d_l = (d[0], u[1], 0)
            u_r = (u[0], d[1], 0)
            aabbs = (d_l, d, u, u_r)
            query_b = [[x_b + aabb[0], y_b + aabb[1], 0] for aabb in aabbs]
            query_e = [[x_e + aabb[0], y_e + aabb[1], 0] for aabb in aabbs]
            query_res = p.rayTestBatch(query_b, query_e)
            # todo: check if actually works, stepSimulation or p.performCollisionDetection()?
            return all([obj == -1 or obj in self.ignore_object_ids for obj, _, _, _, _ in query_res])

    def __str__(self):
        return u'TwoDimRayMotionValidator'


class TwoDimStateValidator(ob.StateValidityChecker, GiskardBehavior):

    def __init__(self, si, collision_checker):
        ob.StateValidityChecker.__init__(self, si)
        GiskardBehavior.__init__(self, str(self))
        self.lock = threading.Lock()
        self.init_map()
        self.init_pybullet_ids_and_joints()
        self.collision_checker = collision_checker
        if not self.pybullet_initialized and not self.map_initialized:
            raise Exception(
                'At least one of the following service must be activated for two dimensional state validation:'
                'import pybullet or start the map server.')

    def init_pybullet_ids_and_joints(self):
        if 'pybullet' in sys.modules:
            self.pybullet_initialized = True
            self.pybullet_joints = []  # todo: remove and use instead self.get_robot().joint_name_to_info[joint_name].joint_index
            for i in range(0, p.getNumJoints(self.get_robot().get_pybullet_id())):
                j = p.getJointInfo(self.get_robot().get_pybullet_id(), i)
                if j[2] != p.JOINT_FIXED:
                    self.pybullet_joints.append(j[1].decode('UTF-8'))
                    # self.rotation_goal = None
        else:
            self.pybullet_initialized = False

    def init_map(self, timeout=3.0):
        try:
            rospy.wait_for_service('static_map', timeout=timeout)
            self.map_initialized = True
        except (rospy.ROSException, rospy.ROSInterruptException) as _:
            rospy.logwarn("Exceeded timeout for map server. Ignoring map...")
            self.map_initialized = False
            return
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
            rospy.logerr("Failed to get static occupancy map. Ignoring map...")
            self.map_initialized = False

    def is_driveable(self, state):
        if self.map_initialized:
            x = numpy.sqrt((state.getX() - self.occ_map_origin.x) ** 2)
            y = numpy.sqrt((state.getY() - self.occ_map_origin.y) ** 2)
            if int(y / self.occ_map_res) >= self.occ_map.shape[0] or \
                    self.occ_map_width - int(x / self.occ_map_res) >= self.occ_map.shape[1]:
                return False
            return 0 <= self.occ_map[int(y / self.occ_map_res)][self.occ_map_width - int(x / self.occ_map_res)] < 1
        else:
            return True

    def is_collision_free(self, state):
        if self.pybullet_initialized:
            x = state.getX()
            y = state.getY()
            # Get current joint states from Bullet and copy them
            old_js = deepcopy(self.get_robot().joint_state)
            robot = self.get_robot()
            # Calc IK for navigating to given state and ...
            poses = [[x, y, 0], [x + 0.1, y, 0], [x - 0.1, y, 0], [x, y + 0.1, 0], [x, y - 0.1, 0]]
            state_iks = list(map(lambda pose:
                                 p.calculateInverseKinematics(robot.get_pybullet_id(),
                                                              robot.get_pybullet_link_id(u'base_footprint'),
                                                              pose), poses))
            # override on current joint states.
            results = []
            for state_ik in state_iks:
                current_js = deepcopy(old_js)
                # Set new joint states in Bullet
                for joint_name, joint_state in zip(self.pybullet_joints, state_ik):
                    current_js[joint_name].position = joint_state
                self.get_robot().joint_state = current_js
                # Check if kitchen is colliding with robot
                results.append(self.collision_checker())
                # Reset joint state
                self.get_robot().joint_state = old_js
            return all(results)
        else:
            return True

    def isValid(self, state):
        # Some arbitrary condition on the state (note that thanks to
        # dynamic type checking we can just call getX() and do not need
        # to convert state to an SE2State.)
        with self.lock:
            return self.is_collision_free(state) and self.is_driveable(state)

    def __str__(self):
        return u'TwoDimStateValidator'


class GlobalPlanner(GetGoal):

    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)

        self.kitchen_space = self.create_kitchen_space()
        self.kitchen_floor_space = self.create_kitchen_floor_space()

        self.pose_goal = None
        self.goal_dict = None

        self.fast_navigation = False

        self.setupNavigation()

    def get_cart_goal(self, cmd):
        try:
            return next(c for c in cmd.constraints if c.type == "CartesianPose")
        except StopIteration:
            return None

    def update_collisions_environment(self):
        self.get_god_map().get_data(identifier.tree_manager).get_node(u'coll').initialise()

    def update_collision_checker(self):
        self.get_god_map().get_data(identifier.tree_manager).get_node(u'coll').update()

    def is_robot_external_collision_free(self):
        self.update_collision_checker()
        closest_points = self.god_map.get_data(identifier.closest_point)
        collisions = []
        for joint_name in self.get_robot().get_link_names():
            if joint_name in closest_points.external_collision:
                collisions.append(closest_points.get_external_collisions(joint_name))
        return len(collisions) == 0

    def is_global_path_needed(self, root_link, tip_link):
        """
        (disclaimer: for correct format please see the source code)

        Returns whether a global path is needed by checking if the shortest path to the goal
        is free of collisions.

        start        new_start (calculated with vector v)
        XXXXXXXXXXXX   ^
        XXcollXobjXX   |  v
        XXXXXXXXXXXX   |
                     goal

        The vector from start to goal in 2D collides with collision object, but the vector
        from new_start to goal does not collide with the environment, but...

        start        new_start
        XXXXXXXXXXXXXXXXXXXXXX
        XXcollisionXobjectXXXX
        XXXXXXXXXXXXXXXXXXXXXX
                     goal

        ... here the shortest path to goal is in collision. Therefore, a global path
        is needed.

        :type root_link: str
        :type tip_link: str
        :rtype: boolean
        """
        curr_R_pose = self.get_robot().get_fk_pose(root_link, tip_link)
        curr_pos = transform_pose(self.get_god_map().get_data(identifier.map_frame), curr_R_pose).pose.position
        curr_arr = numpy.array([curr_pos.x, curr_pos.y, curr_pos.z])
        goal_pos = self.pose_goal.pose.position
        goal_arr = numpy.array([goal_pos.x, goal_pos.y, goal_pos.z])
        obj_id, _, _, _, normal = p.rayTest(curr_arr, goal_arr)[0]
        if obj_id != -1:
            diff = curr_arr - goal_arr
            v = numpy.array(list(normal)) * diff
            new_curr_arr = goal_arr + v
            obj_id, _, _, _, _ = p.rayTest(new_curr_arr, goal_arr)[0]
            return obj_id != -1
        else:
            return False

    def is_global_navigation_needed(self):
        return self.goal_dict[u'tip_link'] == u'base_footprint' and \
               self.goal_dict[u'root_link'] == self.get_robot().get_root() and \
               self.is_global_path_needed(self.goal_dict[u'root_link'], self.goal_dict[u'tip_link'])

    def update(self):

        # Check if move_cmd exists
        move_cmd = self.get_god_map().get_data(identifier.next_move_goal)  # type: MoveCmd
        if not move_cmd:
            return Status.SUCCESS

        # Check if move_cmd contains a Cartesian Goal
        cart_c = self.get_cart_goal(move_cmd)
        if not cart_c:
            return Status.SUCCESS

        # Parse the Cartesian Goal Constraint
        self.goal_dict = yaml.load(cart_c.parameter_value_pair)
        ros_pose = convert_dictionary_to_ros_message(self.goal_dict[u'goal'])
        self.pose_goal = transform_pose(self.get_god_map().get_data(identifier.map_frame), ros_pose)

        if self.is_global_navigation_needed():
            self.update_collisions_environment()
            trajectory = self.planNavigation()
            if trajectory is None or not trajectory.any():
                return Status.FAILURE
            poses = []
            for i, point in enumerate(trajectory):
                if i == 0:
                    continue  # we do not to reach the first pose, since it is the start pose
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
        c = Constraint()
        c.type = u'CartesianPathCarrot'
        c.parameter_value_pair = json.dumps(d[u'parameter_value_pair'])
        return c

    def create_kitchen_space(self):
        # create an SE3 state space
        space = ob.SE3StateSpace()

        # set lower and upper bounds
        bounds = ob.RealVectorBounds(3)
        bounds.setLow(0)
        bounds.setHigh(3)
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

    def get_navigation_planner(self, si):
        # Navigation:
        # RRTstar, RRTsharp: very sparse(= path length eq max range), high number of fp, slow relative to RRTConnect
        # RRTConnect: many points, low number of fp
        # PRMstar: no set_range function, finds no solutions
        # ABITstar: slow, sparse, but good
        if self.fast_navigation:
            planner = og.RRTConnect(si)
            planner.setRange(0.1)
        else:
            planner = og.ABITstar(si)
        # planner.setIntermediateStates(True)
        # planner.setup()
        return planner

    def setupNavigation(self):

        # create a simple setup object
        self.navigation_setup = og.SimpleSetup(self.kitchen_floor_space)

        # Set two dimensional motion and state validator
        si = self.navigation_setup.getSpaceInformation()
        si.setMotionValidator(TwoDimRayMotionValidator(si, self.get_robot()))
        si.setStateValidityChecker(TwoDimStateValidator(si, self.is_robot_external_collision_free))
        si.setup()

        # si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(self.allocOBValidStateSampler))
        # si.setStateValidityCheckingResolution(0.001)

        # Set navigation planner
        planner = self.get_navigation_planner(si)
        self.navigation_setup.setPlanner(planner)

    def planNavigation(self, initial_solve_time=60, refine_solve_time=5,
                       max_num_of_tries=3, max_refine_iterations=0, min_refine_thresh=0.1,
                       plot=True):

        start = self.get_translation_start_state(self.kitchen_floor_space)
        goal = self.get_translation_goal_state(self.kitchen_floor_space)
        self.navigation_setup.setStartAndGoalStates(start, goal)

        optimization_objective = PathLengthAndGoalOptimizationObjective(self.navigation_setup.getSpaceInformation(), goal)
        self.navigation_setup.setOptimizationObjective(optimization_objective)

        planner_status = ob.PlannerStatus(ob.PlannerStatus.UNKNOWN)
        num_try = 0
        solve_time = initial_solve_time
        # Find solution
        while num_try < max_num_of_tries and planner_status.getStatus() not in [ob.PlannerStatus.APPROXIMATE_SOLUTION,
                                                                                ob.PlannerStatus.EXACT_SOLUTION]:
            planner_status = self.navigation_setup.solve(solve_time)
            num_try += 1
        # Refine solution
        refine_iteration = 0
        v_min = 1e6
        while v_min > min_refine_thresh and refine_iteration < max_refine_iterations:
            if not self.fast_navigation:
                v_before = self.navigation_setup.getPlanner().bestCost().value()
            self.navigation_setup.solve(refine_solve_time)
            if not self.fast_navigation:
                v_after = self.navigation_setup.getPlanner().bestCost().value()
                v_min = v_before - v_after
            refine_iteration += 1

        # og.PathSimplifier(si).smoothBSpline(ss.getSolutionPath()) # takes around 20-30 secs with RTTConnect(0.1)
        # og.PathSimplifier(si).reduceVertices(ss.getSolutionPath())
        # Make sure enough subpaths are available for Path Following
        self.navigation_setup.getSolutionPath().interpolate(20)
        if planner_status:
            # try to shorten the path
            # ss.simplifySolution() DONT! NO! DONT UNCOMMENT THAT! NO! STOP IT! FIRST IMPLEMENT CHECKMOTION! THEN TRY AGAIN!
            # print the simplified path
            data = states_matrix_str2array_floats(self.navigation_setup.getSolutionPath().printAsMatrix())  # [[x, y, theta]]
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


def is_3D(space):
    return type(space) == type(ob.SE3StateSpace())


def states_matrix_str2array_floats(str: str, line_sep='\n', float_sep=' '):
    states_strings = str.split(line_sep)
    while '' in states_strings:
        states_strings.remove('')
    return numpy.array(list(map(lambda x: numpy.fromstring(x, dtype=float, sep=float_sep), states_strings)))

#!/usr/bin/env python

import json
import sys
import threading
from collections import namedtuple
from queue import Queue

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
import giskardpy.model.pybullet_wrapper as pw
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.tree.plugin_action_server import GetGoal
from giskardpy.utils.tfwrapper import transform_pose, lookup_pose

from mpl_toolkits.mplot3d import Axes3D
import numpy
import matplotlib.pyplot as plt
from ompl import base as ob
from ompl import geometric as og

# todo: put below in ros params
SolveParameters = namedtuple('SolveParameters', 'initial_solve_time refine_solve_time max_initial_iterations '
                                                'max_refine_iterations min_refine_thresh')
CollisionInfo = namedtuple('CollisionInfo', 'link d u')

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


class TwoDimRayMotionValidator(ob.MotionValidator, GiskardBehavior): #TODO: use collision shapes of links with fixed joints inbetween

    def __init__(self, si, is_3D, object_in_motion, tip_link=None):
        ob.MotionValidator.__init__(self, si)
        GiskardBehavior.__init__(self, str(self))
        if 'pybullet' not in sys.modules:
            raise Exception('For two dimensional motion validation the python module pybullet is needed.')
        self.lock = threading.Lock()
        self.tip_link = tip_link
        self.object_in_motion = object_in_motion
        self.is_3D = is_3D
        self.collision_infos = list()
        self.init_ignore_objects()
        self.update_collision_shape()

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

    def update_tip_link(self, tip_link):
        self.tip_link = tip_link
        self.update_collision_shape()

    def get_collision_info(self, link):
        if self.object_in_motion.has_link_collision(link):
            obj_id = self.object_in_motion.get_pybullet_id()
            link_id = pw.get_link_id(obj_id, link)
            cur_pose = self.object_in_motion.get_fk_pose(self.object_in_motion.get_root(), link)
            cur_pos = cur_pose.pose.position
            aabbs = p.getAABB(obj_id, link_id)
            aabbs_ind = [[aabb[0] - cur_pos.x, aabb[1] - cur_pos.y, aabb[2] - cur_pos.z] for aabb in aabbs]
            return CollisionInfo(link, aabbs_ind[0], aabbs_ind[1])

    def add_first_collision_info(self, joint_names, children_of_tip_link):
        joint_names_r = joint_names if children_of_tip_link else reversed(joint_names)
        for j_n in joint_names_r:
            if children_of_tip_link:
                link_name = self.object_in_motion.get_child_link_of_joint(j_n)
            else:
                link_name = self.object_in_motion.get_parent_link_of_joint(j_n)
            if self.object_in_motion.has_link_collision(link_name):
                self.collision_infos.append(self.get_collision_info(link_name))
                return True
        return False

    def search_childs_for_collision_information(self, start_link, assume_fixed_joints=False):
        queue = Queue()
        queue.put(start_link)
        visited = [start_link]

        while not queue.empty():
            link_popped = queue.get()

            if link_popped != start_link:
                rospy.logwarn(u'Breadth-First searching for collision information'
                              u' further the childs of {}.'.format(start_link))

            children_joint_names = self.object_in_motion.get_child_joints_of_link(link_popped)
            if children_joint_names is not None:
                fixed_children_joint_names = [c_n for c_n in children_joint_names
                                              if self.object_in_motion.is_joint_fixed(c_n) or assume_fixed_joints]
                added = self.add_first_collision_info(fixed_children_joint_names, children_of_tip_link=True)
                if added:
                    return True
                else:
                    children_link_names = [self.object_in_motion.get_child_link_of_joint(c_n)
                                           for c_n in children_joint_names]
                    for link_name in children_link_names:
                        if link_name not in visited:
                            visited.append(link_name)
                            queue.put(link_name)
        return False

    def update_collision_shape(self):
        self.collision_infos = list()
        if self.tip_link is not None:
            tip_link_interaction = self.object_in_motion.get_child_links_of_link(self.tip_link) is None
            # If tip link is not used for interaction and has collision information,
            # we use only this information for raytesting, otherwise...
            if not tip_link_interaction:
                ci = self.get_collision_info(self.tip_link)
                if ci is not None:
                    self.collision_infos.append(ci)
                    return True

            # ... we check the parent links of the tip link for fixed links and their collision information.
            chain_joint_names = self.object_in_motion.get_joint_names_from_chain(self.object_in_motion.get_root(), self.tip_link)
            fixed_joint_names = [j_n for j_n in chain_joint_names if self.object_in_motion.is_joint_fixed(j_n)]
            added = self.add_first_collision_info(fixed_joint_names, children_of_tip_link=False)
            # If there were no fixed parent links with collision information added,
            # we search for collision information in the childs links if possible.
            if not added and not tip_link_interaction:
                self.search_childs_for_collision_information(self.tip_link)
            # Further we model interaction tip_links such that it makes sense to search,
            # for neighboring links and their collision information. Since the joints
            # of these links are normally not fixed, we assume that they are.
            if tip_link_interaction:
                parent = self.object_in_motion.get_parent_link_of_link(self.tip_link)
                self.search_childs_for_collision_information(parent, assume_fixed_joints=True)
        return len(self.collision_infos) > 0

    def aabb_to_3d_points(self, d, u):
        d_new = [d[0], d[1], u[2]]
        u_new = [u[0], u[1], d[2]]
        bottom_cube_face = self.aabb_to_2d_points(d, u_new)
        upper_cube_face = self.aabb_to_2d_points(d_new, u)
        res = []
        for p in bottom_cube_face:
            res.append([p[0], p[1], d[2]])
        for p in upper_cube_face:
            res.append([p[0], p[1], u[2]])
        return res

    def aabb_to_2d_points(self, d, u):
        d_l = [d[0], u[1], 0]
        u_r = [u[0], d[1], 0]
        return [d_l, d, u, u_r]

    def checkMotion(self, s1, s2):
        # checkMotion calls checkMotion with Map anc checkMotion with Bullet
        if self.tip_link is None:
            raise Exception(u'Please set tip_link for {}.'.format(str(self)))
        with self.lock:
            x_b = s1.getX()
            y_b = s1.getY()
            x_e = s2.getX()
            y_e = s2.getY()
            if self.is_3D:
                z_b = s1.getZ()
                z_e = s2.getZ()
            # Shoot ray from start to end pose and check if it intersects with the kitchen,
            # if so return false, else true.
            # TODO: for each loop for every collision info
            for collision_info in self.collision_infos:
                if self.is_3D:
                    aabbs = self.aabb_to_3d_points(collision_info.d, collision_info.u)
                    query_b = [[x_b + aabb[0], y_b + aabb[1], z_b + aabb[2]] for aabb in aabbs]
                    query_e = [[x_e + aabb[0], y_e + aabb[1], z_e + aabb[2]] for aabb in aabbs]
                else:
                    aabbs = self.aabb_to_2d_points(collision_info.d, collision_info.u)
                    query_b = [[x_b + aabb[0], y_b + aabb[1], 0] for aabb in aabbs]
                    query_e = [[x_e + aabb[0], y_e + aabb[1], 0] for aabb in aabbs]
                # If the aabb box is not from the self.tip_link, add the translation from the
                # self.tip_link to the collision info link.
                if collision_info.link != self.tip_link:
                    tip_link_2_coll_link_pose = lookup_pose(self.tip_link, collision_info.link)
                    point = tip_link_2_coll_link_pose.pose.position
                    query_b = [[point.x + q_b[0], point.y + q_b[1], point.z + q_b[2]] for q_b in deepcopy(query_b)]
                    query_e = [[point.x + q_e[0], point.y + q_e[1], point.z + q_e[2]] for q_e in deepcopy(query_e)]
                query_res = p.rayTestBatch(query_b, query_e)
                collision_free = all([obj == -1 or obj in self.ignore_object_ids for obj, _, _, _, _ in query_res])
                if not collision_free:
                    return False
            return True

    def __str__(self):
        return u'TwoDimRayMotionValidator'


class BulletCollisionChecker(GiskardBehavior):

    def __init__(self, is_3D, collision_checker, tip_link=None):
        GiskardBehavior.__init__(self, str(self))
        self.lock = threading.Lock()
        self.init_pybullet_ids_and_joints()
        self.collision_checker = collision_checker
        self.is_3D = is_3D
        self.tip_link = tip_link

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

    def is_collision_free(self, state):
        if self.tip_link is None:
            raise Exception(u'Please set tip_link for {}.'.format(str(self)))
        if self.pybullet_initialized:
            x = state.getX()
            y = state.getY()
            if self.is_3D:
                z = state.getZ()
            # Get current joint states from Bullet and copy them
            old_js = deepcopy(self.get_robot().joint_state)
            robot = self.get_robot()
            # Calc IK for navigating to given state and ...
            if self.is_3D:
                poses = [[x, y, z], [x + 0.1, y, z], [x - 0.1, y, z], [x, y + 0.1, z], [x, y - 0.1, z]]
            else:
                poses = [[x, y, 0], [x + 0.1, y, 0], [x - 0.1, y, 0], [x, y + 0.1, 0], [x, y - 0.1, 0]]
            state_iks = list(map(lambda pose:
                                 p.calculateInverseKinematics(robot.get_pybullet_id(),
                                                              robot.get_pybullet_link_id(self.tip_link),
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

class ThreeDimStateValidator(ob.StateValidityChecker):

    def __init__(self, si, collision_checker):
        ob.StateValidityChecker.__init__(self, si)
        self.collision_checker = collision_checker
        self.lock = threading.Lock()
        if not collision_checker.pybullet_initialized:
            raise Exception('Please import pybullet.')

    def update(self, tip_link):
        self.collision_checker.tip_link = tip_link

    def isValid(self, state):
        with self.lock:
            return self.collision_checker.is_collision_free(state)

class TwoDimStateValidator(ob.StateValidityChecker):

    def __init__(self, si, collision_checker):
        ob.StateValidityChecker.__init__(self, si)
        self.collision_checker = collision_checker
        self.lock = threading.Lock()
        self.init_map()
        if not self.collision_checker.pybullet_initialized and not self.map_initialized:
            raise Exception(
                'At least one of the following service must be activated for two dimensional state validation:'
                'import pybullet or start the map server.')

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

    def isValid(self, state):
        # Some arbitrary condition on the state (note that thanks to
        # dynamic type checking we can just call getX() and do not need
        # to convert state to an SE2State.)
        with self.lock:
            return self.collision_checker.is_collision_free(state) and self.is_driveable(state)

    def __str__(self):
        return u'TwoDimStateValidator'


class GlobalPlanner(GetGoal):

    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)

        self.kitchen_space = self.create_kitchen_space()
        self.kitchen_floor_space = self.create_kitchen_floor_space()

        self.pose_goal = None
        self.__goal_dict = None

        self._planner_solve_params = {}  # todo: load values from rosparam
        self.navigation_config = 'fast_without_refine'  # todo: load value from rosparam
        self.movement_config = 'fast_without_refine'

        self.setupNavigation()
        self.setupMovement()

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

    def is_global_path_needed(self):
        return self.__is_global_path_needed(self.get_world().get_objects()['kitchen']._pybullet_id)

    def __is_global_path_needed(self, coll_obj_id):
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

        :rtype: boolean
        """
        curr_R_pose = self.get_robot().get_fk_pose(self.root_link, self.tip_link)
        curr_pos = transform_pose(self.get_god_map().get_data(identifier.map_frame), curr_R_pose).pose.position
        curr_arr = numpy.array([curr_pos.x, curr_pos.y, curr_pos.z])
        goal_pos = self.pose_goal.pose.position
        goal_arr = numpy.array([goal_pos.x, goal_pos.y, goal_pos.z])
        obj_id, _, _, _, normal = p.rayTest(curr_arr, goal_arr)[0]
        if obj_id == coll_obj_id:
            diff = numpy.sqrt(numpy.sum((curr_arr - goal_arr) ** 2))
            v = numpy.array(list(normal)) * diff
            new_curr_arr = goal_arr + v
            obj_id, _, _, _, _ = p.rayTest(new_curr_arr, goal_arr)[0]
            return obj_id == coll_obj_id
        else:
            return False

    def is_global_navigation_needed(self):
        return self.tip_link == u'base_footprint' and \
               self.root_link == self.get_robot().get_root() and \
               self.is_global_path_needed()

    def save_cart_goal(self, cart_c):

        self.__goal_dict = yaml.load(cart_c.parameter_value_pair)
        ros_pose = convert_dictionary_to_ros_message(self.__goal_dict[u'goal'])
        self.pose_goal = transform_pose(self.get_god_map().get_data(identifier.map_frame), ros_pose)

        self.root_link = self.__goal_dict[u'root_link']
        self.tip_link = self.__goal_dict[u'tip_link']
        link_names = self.get_robot().get_link_names()

        if self.root_link not in link_names:
            raise Exception(u'Root_link {} is no known link of the robot.'.format(self.root_link))
        if self.tip_link not in link_names:
            raise Exception(u'Tip_link {} is no known link of the robot.'.format(self.tip_link))
        if not self.get_robot().get_link_names_from_chain(self.root_link, self.tip_link):
            raise Exception(u'Did not found link chain of the robot from'
                            u' root_link {} to tip_link {}.'.format(self.root_link, self.tip_link))

    def update(self):

        # Check if move_cmd exists
        move_cmd = self.get_god_map().get_data(identifier.next_move_goal)  # type: MoveCmd
        if not move_cmd:
            return Status.SUCCESS

        # Check if move_cmd contains a Cartesian Goal
        cart_c = self.get_cart_goal(move_cmd)
        if not cart_c:
            return Status.SUCCESS

        # Parse and save the Cartesian Goal Constraint
        self.save_cart_goal(cart_c)

        if self.is_global_navigation_needed():
            self.update_collisions_environment()
            trajectory = self.planNavigation()
        elif self.is_global_path_needed():
            self.update_collisions_environment()
            trajectory = self.planMovement()
        else:
            return Status.SUCCESS

        if len(trajectory) == 0:
            return Status.FAILURE
        poses = []
        for i, point in enumerate(trajectory):
            if i == 0:
                continue  # important assumption for constraint:
                # we do not to reach the first pose, since it is the start pose
            base_pose = PoseStamped()
            base_pose.header.frame_id = self.get_god_map().get_data(identifier.map_frame)
            base_pose.pose.position.x = point[0]
            base_pose.pose.position.y = point[1]
            base_pose.pose.position.z = point[2] if len(point) > 3 else 0
            base_pose.pose.orientation = Quaternion(*quaternion_about_axis(0, [0, 0, 1]))
            poses.append(base_pose)
        poses[-1].pose.orientation = self.pose_goal.pose.orientation
        move_cmd.constraints = self.get_cartesian_path_constraints(poses)
        self.get_god_map().set_data(identifier.next_move_goal, move_cmd)
        return Status.SUCCESS

    def get_cartesian_path_constraints(self, poses):

        d = dict()
        d[u'parameter_value_pair'] = deepcopy(self.__goal_dict)
        d[u'parameter_value_pair'].pop(u'goal')

        oc_d = deepcopy(d)
        oc_d[u'parameter_value_pair'][u'goal'] = convert_ros_message_to_dictionary(poses[-1])
        oc = Constraint()
        oc.type = u'CartesianOrientation'
        oc.parameter_value_pair = json.dumps(oc_d[u'parameter_value_pair'])

        c_d = deepcopy(d)
        c_d[u'parameter_value_pair'][u'goals'] = list(map(convert_ros_message_to_dictionary, poses))
        c = Constraint()
        c.type = u'CartesianPathCarrot'
        c.parameter_value_pair = json.dumps(c_d[u'parameter_value_pair'])

        return [c, oc]

    def create_kitchen_space(self):
        # create an SE3 state space
        space = ob.SE3StateSpace()

        # set lower and upper bounds
        bounds = ob.RealVectorBounds(3)
        bounds.setLow(-15)
        bounds.setHigh(15)
        bounds.setLow(2, 0)
        bounds.setHigh(2, 5)
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

    def get_translation_start_state(self, space, round_decimal_place=3):
        state = ob.State(space)
        pose_root_linkTtip_link = self.get_robot().get_fk_pose(self.root_link, self.tip_link)
        pose_mTtip_link = transform_pose(self.get_god_map().get_data(identifier.map_frame), pose_root_linkTtip_link)
        state().setX(round(pose_mTtip_link.pose.position.x, round_decimal_place))
        state().setY(round(pose_mTtip_link.pose.position.y, round_decimal_place))
        if is_3D(space):
            state().setZ(round(pose_mTtip_link.pose.position.z, round_decimal_place))
        return state

    def get_translation_goal_state(self, space, round_decimal_place=3):
        state = ob.State(space)
        state().setX(round(self.pose_goal.pose.position.x, round_decimal_place))
        state().setY(round(self.pose_goal.pose.position.y, round_decimal_place))
        if is_3D(space):
            state().setZ(round(self.pose_goal.pose.position.z, round_decimal_place))
        return state

    def allocOBValidStateSampler(si):
        return ob.GaussianValidStateSampler(si)

    def get_navigation_planner(self, si):
        # Navigation:
        # RRTstar, RRTsharp: very sparse(= path length eq max range), high number of fp, slow relative to RRTConnect
        # RRTConnect: many points, low number of fp
        # PRMstar: no set_range function, finds no solutions
        # ABITstar: slow, sparse, but good
        # if self.fast_navigation:
        #    planner = og.RRTConnect(si)
        #    planner.setRange(0.1)
        # else:
        planner = og.ABITstar(si)
        self._planner_solve_params[planner.getName()] = {
            'slow_without_refine': SolveParameters(initial_solve_time=60, refine_solve_time=5, max_initial_iterations=3,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'slow_with_refine': SolveParameters(initial_solve_time=60, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5),
            'fast_without_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'fast_with_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5)
        }
        # planner.setIntermediateStates(True)
        # planner.setup()
        return planner

    def setupNavigation(self):

        # create a simple setup object
        self.navigation_setup = og.SimpleSetup(self.kitchen_floor_space)

        # Set two dimensional motion and state validator
        si = self.navigation_setup.getSpaceInformation()
        si.setMotionValidator(TwoDimRayMotionValidator(si, False, self.get_robot(), tip_link=u'base_footprint'))
        collision_checker = BulletCollisionChecker(False, self.is_robot_external_collision_free,
                                                   tip_link=u'base_footprint')
        si.setStateValidityChecker(TwoDimStateValidator(si, collision_checker))
        si.setup()

        # si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(self.allocOBValidStateSampler))
        # si.setStateValidityCheckingResolution(0.001)

        # Set navigation planner
        planner = self.get_navigation_planner(si)
        self.navigation_setup.setPlanner(planner)

    def setupMovement(self):

        # create a simple setup object
        self.movement_setup = og.SimpleSetup(self.kitchen_space)

        # Set two dimensional motion and state validator
        si = self.movement_setup.getSpaceInformation()
        si.setMotionValidator(TwoDimRayMotionValidator(si, True, self.get_robot()))
        collision_checker = BulletCollisionChecker(True, self.is_robot_external_collision_free)
        si.setStateValidityChecker(ThreeDimStateValidator(si, collision_checker))
        si.setup()

        # si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(self.allocOBValidStateSampler))
        # si.setStateValidityCheckingResolution(0.001)

        # Set navigation planner
        planner = self.get_navigation_planner(si)
        self.movement_setup.setPlanner(planner)

    def planNavigation(self, plot=True):

        # Set Start and Goal pose for Planner
        start = self.get_translation_start_state(self.kitchen_floor_space)
        goal = self.get_translation_goal_state(self.kitchen_floor_space)
        self.navigation_setup.setStartAndGoalStates(start, goal)

        # Set Optimization Function
        optimization_objective = PathLengthAndGoalOptimizationObjective(self.navigation_setup.getSpaceInformation(), goal)
        self.navigation_setup.setOptimizationObjective(optimization_objective)

        planner_status = self.solve(self.navigation_setup, self.navigation_config)
        # og.PathSimplifier(si).smoothBSpline(ss.getSolutionPath()) # takes around 20-30 secs with RTTConnect(0.1)
        # og.PathSimplifier(si).reduceVertices(ss.getSolutionPath())
        data = numpy.array([])
        if planner_status in [ob.PlannerStatus.APPROXIMATE_SOLUTION, ob.PlannerStatus.EXACT_SOLUTION]:
            # try to shorten the path
            # ss.simplifySolution() DONT! NO! DONT UNCOMMENT THAT! NO! STOP IT! FIRST IMPLEMENT CHECKMOTION! THEN TRY AGAIN!
            # Make sure enough subpaths are available for Path Following
            self.navigation_setup.getSolutionPath().interpolate(20)
            data = states_matrix_str2array_floats(self.navigation_setup.getSolutionPath().printAsMatrix())  # [[x, y, theta]]
            # print the simplified path
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

    def planMovement(self, plot=True):

        # Update Collision Checker
        self.movement_setup.getSpaceInformation().getStateValidityChecker().update(self.tip_link)
        #self.movement_setup.getSpaceInformation().getMotionValidator().update_tip_link(self.tip_link)
        si = self.movement_setup.getSpaceInformation()
        si.setMotionValidator(TwoDimRayMotionValidator(si, True, self.get_robot(), tip_link=self.tip_link))


        start = self.get_translation_start_state(self.kitchen_space)
        goal = self.get_translation_goal_state(self.kitchen_space)
        # Force goal and start pose into limited search area
        self.kitchen_space.enforceBounds(start())
        self.kitchen_space.enforceBounds(goal())
        self.movement_setup.setStartAndGoalStates(start, goal)

        optimization_objective = PathLengthAndGoalOptimizationObjective(self.movement_setup.getSpaceInformation(), goal)
        self.movement_setup.setOptimizationObjective(optimization_objective)

        planner_status = self.solve(self.movement_setup, self.movement_config)
        # og.PathSimplifier(si).smoothBSpline(ss.getSolutionPath()) # takes around 20-30 secs with RTTConnect(0.1)
        # og.PathSimplifier(si).reduceVertices(ss.getSolutionPath())
        data = numpy.array([])
        if planner_status in [ob.PlannerStatus.APPROXIMATE_SOLUTION, ob.PlannerStatus.EXACT_SOLUTION]:
            # try to shorten the path
            # ss.simplifySolution() DONT! NO! DONT UNCOMMENT THAT! NO! STOP IT! FIRST IMPLEMENT CHECKMOTION! THEN TRY AGAIN!
            # Make sure enough subpaths are available for Path Following
            self.movement_setup.getSolutionPath().interpolate(20)

            data = states_matrix_str2array_floats(self.movement_setup.getSolutionPath().printAsMatrix()) # [x y z xw yw zw w]
            # print the simplified path
            if plot:
                plt.close()
                fig, ax = plt.subplots()
                ax.plot(data[:, 1], data[:, 0])
                ax.invert_xaxis()
                ax.set(xlabel='y (m)', ylabel='x (m)',
                       title='2D Path in map')
                # ax = fig.gca(projection='2d')
                # ax.plot(data[:, 0], data[:, 1], '.-')
                plt.show()
        return data

    def solve(self, setup, config):

        # Get solve parameters
        solve_params = self._planner_solve_params[setup.getPlanner().getName()][config]
        initial_solve_time = solve_params.initial_solve_time
        refine_solve_time = solve_params.refine_solve_time
        max_initial_iterations = solve_params.max_initial_iterations
        max_refine_iterations = solve_params.max_refine_iterations
        min_refine_thresh = solve_params.min_refine_thresh
        max_initial_solve_time = initial_solve_time * max_initial_iterations
        max_refine_solve_time = refine_solve_time * max_refine_iterations

        planner_status = ob.PlannerStatus(ob.PlannerStatus.UNKNOWN)
        num_try = 0
        time_solving_intial = 0
        # Find solution
        while num_try < max_initial_iterations and time_solving_intial < max_initial_solve_time and \
                planner_status.getStatus() not in [ob.PlannerStatus.APPROXIMATE_SOLUTION,
                                                   ob.PlannerStatus.EXACT_SOLUTION]:
            planner_status = setup.solve(initial_solve_time)
            time_solving_intial += setup.getLastPlanComputationTime()
            num_try += 1
        # Refine solution
        refine_iteration = 0
        v_min = 1e6
        time_solving_refine = 0
        if planner_status.getStatus() in [ob.PlannerStatus.APPROXIMATE_SOLUTION, ob.PlannerStatus.EXACT_SOLUTION]:
            while v_min > min_refine_thresh and refine_iteration < max_refine_iterations and \
                    time_solving_refine < max_refine_solve_time:
                if min_refine_thresh is not None:
                    v_before = setup.getPlanner().bestCost().value()
                setup.solve(refine_solve_time)
                time_solving_refine += setup.getLastPlanComputationTime()
                if min_refine_thresh is not None:
                    v_after = setup.getPlanner().bestCost().value()
                    v_min = v_before - v_after
                refine_iteration += 1
        return planner_status.getStatus()


def is_3D(space):
    return type(space) == type(ob.SE3StateSpace())


def states_matrix_str2array_floats(str: str, line_sep='\n', float_sep=' '):
    states_strings = str.split(line_sep)
    while '' in states_strings:
        states_strings.remove('')
    return numpy.array(list(map(lambda x: numpy.fromstring(x, dtype=float, sep=float_sep), states_strings)))

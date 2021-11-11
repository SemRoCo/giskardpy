#!/usr/bin/env python

import json
import sys
import threading
from collections import namedtuple
from queue import Queue

import numpy as np
import rospy
import tf.transformations
import yaml
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from visualization_msgs.msg import MarkerArray

from giskard_msgs.msg import Constraint
from nav_msgs.srv import GetMap
from py_trees import Status
import pybullet as p

from copy import deepcopy
from tf.transformations import quaternion_about_axis

import giskardpy.identifier as identifier
import giskardpy.model.pybullet_wrapper as pw
from giskardpy.model.utils import make_world_body_box
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.tree.get_goal import GetGoal
from giskardpy.utils import tfwrapper
from giskardpy.utils.kdl_parser import KDL
from giskardpy.utils.tfwrapper import transform_pose, lookup_pose, np_to_pose, np_to_pose_stamped, list_to_kdl

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


class CollisionCheckerInterface(GiskardBehavior):

    def __init__(self, name='CollisionCheckerInterface'):
        super().__init__(name)

    def update_collisions_environment(self):
        self.get_god_map().get_data(identifier.tree_manager).get_node(u'collision checker').initialise()
        self.collision_scene.sync()
        rospy.sleep(2.0)

    def update_collision_checker(self):
        self.get_god_map().get_data(identifier.tree_manager).get_node(u'collision checker').update()

    def is_robot_link_external_collision_free(self, link_name):
        self.update_collision_checker()
        closest_points = self.god_map.get_data(identifier.closest_point)
        collisions = []
        if link_name in closest_points.external_collision:
            collisions.append(closest_points.get_external_collisions(link_name))
        return len(collisions) == 0

    def is_robot_external_collision_free(self):
        self.update_collision_checker()
        closest_points = self.god_map.get_data(identifier.closest_point)
        collisions = []
        for link_name in self.get_robot().link_names:
            if link_name in closest_points.external_collision:
                collisions.append(closest_points.get_external_collisions(link_name))
        return len(collisions) == 0

    def is_object_external_collision_free(self, object_name):
        self.update_collision_checker()
        obj = self.get_world().get_object(object_name)
        closest_points = self.god_map.get_data(identifier.closest_point)
        for link_name in obj.get_link_names():
            if link_name in closest_points.external_collision:
                return False
        return True

    def are_objects_external_collision_free(self, ignore_object_names):
        self.update_collision_checker()
        closest_points = self.god_map.get_data(identifier.closest_point)
        for obj in self.get_world().get_objects():
            if obj.get_name() in ignore_object_names:
                continue
            for link_name in obj.get_link_names():
                if link_name in closest_points.external_collision:
                    return False
        return True

    def are_objects_external_collision_free_except_robot(self):
        return self.are_objects_external_collision_free([self.get_robot().get_name()])

    def get_collisions(self, link_name):
        all_collisions = self.get_god_map().get_data(identifier.closest_point).items()
        link_collisions = list()
        for c in all_collisions:
            if c.get_link_b() == link_name or c.get_link_a() == link_name:
                link_collisions.append(c)
        return link_collisions


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


class AbstractMotionValidator(ob.MotionValidator):  # TODO: use collision shapes of links with fixed joints inbetween

    def __init__(self, si, is_3D, object_in_motion, collision_scene, tip_link=None):
        ob.MotionValidator.__init__(self, si)
        self.lock = threading.Lock()
        self.collision_scene = collision_scene
        self.tip_link = tip_link
        self.object_in_motion = object_in_motion
        self.is_3D = is_3D
        self.collision_aabbs = list()
        self.ignore_object_ids = list()
        self.map_frame = 'map'
        self.update()

    def update_ignore_object_ids(self):
        self.ignore_object_ids = [self.collision_scene.object_name_to_bullet_id[link_name]
                                  for link_name in self.object_in_motion.link_names_with_collisions]

    def update(self, tip_link=None):
        if tip_link is not None:
            self.tip_link = tip_link
        if self.tip_link is not None:
            self.update_collision_shape()
            self.update_ignore_object_ids()

    def get_collision_info(self, link_name):
        if self.object_in_motion.has_link_collisions(link_name):
            link_id = self.collision_scene.object_name_to_bullet_id[link_name]
            cur_pose = self.collision_scene.fks[link_name]
            cur_pos = cur_pose[:3]
            aabbs = p.getAABB(link_id)
            aabbs_ind = [aabb - cur_pos for aabb in aabbs]
            return CollisionInfo(link_name, aabbs_ind[0], aabbs_ind[1])

    #def add_first_collision_info(self, joint_names, children_of_tip_link):
    #    joint_names_r = joint_names if children_of_tip_link else reversed(joint_names)
    #    for j_n in joint_names_r:
    #        if children_of_tip_link:
    #            link_name = self.object_in_motion.get_child_link_of_joint(j_n)
    #        else:
    #            link_name = self.object_in_motion.get_parent_link_of_joint(j_n)
    #        if self.object_in_motion.has_link_collisions(link_name):
    #            self.collision_aabbs.append(self.get_collision_info(link_name))
    #            return True
    #    return False

    #def search_childs_for_collision_information(self, start_link, assume_fixed_joints=False):
    #    queue = Queue()
    #    queue.put(start_link)
    #    visited = [start_link]
    #
    #    while not queue.empty():
    #        link_popped = queue.get()
    #
    #        if link_popped != start_link:
    #            rospy.logwarn(u'Breadth-First searching for collision information'
    #                          u' further the children of {}.'.format(start_link))
    #
    #        children_joint_names = self.object_in_motion.get_child_joints_of_link(link_popped)
    #        if children_joint_names is not None:
    #            fixed_children_joint_names = [c_n for c_n in children_joint_names
    #                                          if self.object_in_motion.is_joint_fixed(c_n) or assume_fixed_joints]
    #            added = self.add_first_collision_info(fixed_children_joint_names, children_of_tip_link=True)
    #            if added:
    #                return True
    #            else:
    #                children_link_names = [self.object_in_motion.get_child_link_of_joint(c_n)
    #                                       for c_n in children_joint_names]
    #                for link_name in children_link_names:
    #                    if link_name not in visited:
    #                        visited.append(link_name)
    #                        queue.put(link_name)
    #    return False

    def update_collision_shape(self):
        """
        If tip link is not used for interaction and has collision information, we use only this information for
        continuous collision checking. Otherwise, we check the parent links of the tip link for fixed links and
        their collision information. If there were no fixed parent links with collision information added,
        we search for collision information in the childs links if possible.
        Further we model interaction tip_links such that it makes sense to search,
        for neighboring links and their collision information. Since the joints
        of these links are normally not fixed, we assume that they are.

        """
        self.collision_aabbs = list()
        if self.tip_link is not None:
            tip_joint_name = self.collision_scene.world.links[self.tip_link].parent_joint_name
            tip_link_interaction = not self.object_in_motion.links[self.tip_link].has_children()
            # Get tip link collision and return if possible
            if not tip_link_interaction:
                ci = self.get_collision_info(self.tip_link)
                if ci is not None:
                    self.collision_aabbs.append(ci)
                    return True
            # Get parent or child link collision
            parent_link_name = None
            try:
                parent_link_name = self.object_in_motion.get_parent_link_with_collisions(tip_joint_name)
            except KeyError:
                pass
            if parent_link_name is not None:
                self.collision_aabbs.append(self.get_collision_info(parent_link_name))
            elif not tip_link_interaction:
                child_link_names = self.object_in_motion.get_direct_children_with_collisions(tip_joint_name)
                if len(child_link_names) > 0:
                    self.collision_aabbs.append(self.get_collision_info(child_link_names[0]))
            # Add siblings if tip has no children
            if tip_link_interaction:
                sibling_link_names = self.object_in_motion.get_siblings_with_collisions(tip_joint_name)
                if len(sibling_link_names) > 0:
                    self.collision_aabbs.append(self.get_collision_info(sibling_link_names[0]))
        return len(self.collision_aabbs) > 0

    def update_collision_shape_new(self):
        """
        If tip link is not used for interaction and has collision information, we use only this information for
        continuous collision checking. Otherwise, we check the parent links of the tip link for fixed links and
        their collision information. If there were no fixed parent links with collision information added,
        we search for collision information in the childs links if possible.
        Further we model interaction tip_links such that it makes sense to search,
        for neighboring links and their collision information. Since the joints
        of these links are normally not fixed, we assume that they are.

        """
        self.collision_aabbs = list()
        if self.tip_link is not None:
            controlled_parent_joint_name = self.object_in_motion.get_controlled_parent_joint_of_link(self.tip_link)
            child_links_with_collision, _ = self.object_in_motion.search_branch(controlled_parent_joint_name,
                                                                                collect_link_when=self.object_in_motion.has_link_collisions)
            for link_name in child_links_with_collision:
                self.collision_aabbs.append(self.get_collision_info(link_name))
            return len(self.collision_aabbs) > 0

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
        raise Exception('Not implemented.')

    def __str__(self):
        return u'RayMotionValidator'


class RayMotionValidator(AbstractMotionValidator):

    def __init__(self, si, is_3D, object_in_motion, collision_scene, tip_link=None):
        super(RayMotionValidator, self).__init__(si, is_3D, object_in_motion, collision_scene, tip_link)

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
            for collision_info in self.collision_aabbs:
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
                    map_2_tip_link_pos = lookup_pose(self.map_frame, self.tip_link).pose.position
                    map_2_coll_link_pos = lookup_pose(self.map_frame, collision_info.link).pose.position
                    point = [map_2_coll_link_pos.x - map_2_tip_link_pos.x,
                             map_2_coll_link_pos.y - map_2_tip_link_pos.y,
                             map_2_coll_link_pos.z - map_2_tip_link_pos.z]
                    query_b = [[point[0] + q_b[0], point[1] + q_b[1], point[2] + q_b[2]] for q_b in deepcopy(query_b)]
                    query_e = [[point[0] + q_e[0], point[1] + q_e[1], point[2] + q_e[2]] for q_e in deepcopy(query_e)]
                query_res = p.rayTestBatch(query_b, query_e)
                collision_free = all([obj == -1 or obj in self.ignore_object_ids for obj, _, _, _, _ in query_res])
                if not collision_free:
                    return False
            return True


class CompoundBoxMotionValidator(AbstractMotionValidator):

    def __init__(self, si, is_3D, object_in_motion, tip_link=None):
        super(CompoundBoxMotionValidator, self).__init__(si, is_3D, object_in_motion, tip_link)

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
                    start_and_end_positions = [[x_b, y_b, z_b], [x_e, y_e, z_e]]
                else:
                    start_and_end_positions = [[x_b, y_b, 0.001], [x_e, y_e, 0.001]]
                min_size = np.max(np.abs(np.array(collision_info.d) - np.array(collision_info.u)))
                # If the aabb box is not from the self.tip_link, add the translation from the
                # self.tip_link to the collision info link.
                if collision_info.link != self.tip_link:
                    tip_link_2_coll_link_pose = lookup_pose(self.tip_link, collision_info.link)
                    point = tip_link_2_coll_link_pose.pose.position
                    start_and_end_positions = [[point.x + q_b[0], point.y + q_b[1], point.z + q_b[2]] \
                                               for q_b in deepcopy(start_and_end_positions)]
                box_space = CompoundBoxSpace(self.get_world(), self.get_robot(),
                                             self.get_god_map().get_data(identifier.map_frame),
                                             CollisionCheckerInterface(), min_size, start_and_end_positions)
                if len(box_space.get_collisions().values()) != 0:
                    return False
            return True

class RobotBulletCollisionChecker(GiskardBehavior):

    def __init__(self, is_3D, collision_checker=None, tip_link=None, collision_offset=0.1):
        GiskardBehavior.__init__(self, str(self))
        self.lock = threading.Lock()
        self.robot_kdl_tree = KDL(rospy.get_param('robot_description'))
        if collision_checker is None:
            self._collision_checker = CollisionCheckerInterface()
            self.collision_checker = self._collision_checker.is_robot_link_external_collision_free
        else:
            self.collision_checker = collision_checker
        self.is_3D = is_3D
        self.tip_link = tip_link
        self.collision_offset = collision_offset

    def get_ik(self, pose):
        old_js = self.get_god_map().get_data(identifier.joint_states)
        new_js = deepcopy(old_js)
        js_dict_position = {}
        for k, v in old_js.items():
            js_dict_position[k] = v.position
        kdl_robot = self.robot_kdl_tree.get_robot(self.robot.root_link, self.tip_link)
        joint_array = kdl_robot.ik(js_dict_position, list_to_kdl(pose))
        for i, joint_name in enumerate(kdl_robot.joints):
            new_js[joint_name].position = joint_array[i]
        return new_js

    def is_collision_free(self, state):
        if self.tip_link is None:
            raise Exception(u'Please set tip_link for {}.'.format(str(self)))
        if True:
            x = state.getX()
            y = state.getY()
            if self.is_3D:
                z = state.getZ()
                r = state.rotation()
                rot = [r.x, r.y, r.z, r.w]
            else:
                yaw = state.getYaw()
                rot = p.getQuaternionFromEuler(
                    [0, 0, yaw])  # todo: make configurable to ignore rotation, since navigation works better without it
            # Get current joint states
            old_js = self.get_god_map().get_data(identifier.joint_states)
            # Calc IK for navigating to given state and ...
            if self.is_3D:
                poses = [[[x, y, z], rot], [[x + self.collision_offset, y, z], rot],
                         [[x - self.collision_offset, y, z], rot],
                         [[x, y + self.collision_offset, z], rot], [[x, y - self.collision_offset, z], rot]]
            else:
                poses = [[[x, y, 0], rot], [[x + self.collision_offset, y, 0], rot],
                         [[x - self.collision_offset, y, 0], rot],
                         [[x, y + self.collision_offset, 0], rot], [[x, y - self.collision_offset, 0], rot]]
            state_iks = [self.get_ik(pose) for pose in poses]
            # override on current joint states.
            results = []
            for state_ik in state_iks:
                self.get_robot().state = state_ik
                #self.get_god_map().get_data(identifier.tree_manager).get_node(u'visualization').initialise()
                #self.get_god_map().get_data(identifier.tree_manager).get_node(u'visualization').update()
                # Check if kitchen is colliding with robot
                results.append(self.collision_checker(self.tip_link))
                # Reset joint state
                self.get_robot().state = old_js
            return all(results)
        else:
            return True


class PyBulletWorldObjectCollisionChecker(GiskardBehavior):

    def __init__(self, is_3D, collision_checker, collision_object_name, collision_offset=0.1):
        GiskardBehavior.__init__(self, str(self))
        self.lock = threading.Lock()
        self.init_pybullet_ids_and_joints()
        self.collision_checker = collision_checker
        self.is_3D = is_3D
        self.collision_object_name = collision_object_name
        self.collision_offset = collision_offset

    def update_object_pose(self, p: Point, q: Quaternion):
        try:
            obj = self.get_world().get_object(self.collision_object_name)
        except KeyError:
            raise Exception(u'Could not find object with name {}.'.format(self.collision_object_name))
        obj.base_pose = Pose(p, q)

    def init_pybullet_ids_and_joints(self):
        if 'pybullet' in sys.modules:
            self.pybullet_initialized = True
        else:
            self.pybullet_initialized = False

    def is_collision_free_ompl(self, state):
        if self.collision_object_name is None:
            raise Exception(u'Please set object for {}.'.format(str(self)))
        if self.pybullet_initialized:
            x = state.getX()
            y = state.getY()
            if self.is_3D:
                z = state.getZ()
                r = state.rotation()
                rot = [r.x, r.y, r.z, r.w]
            else:
                z = 0
                yaw = state.getYaw()
                rot = p.getQuaternionFromEuler([0, 0, yaw])
            # update pose
            self.update_object_pose(Point(x, y, z), Quaternion(rot[0], rot[1], rot[2], rot[3]))
            return self.collision_checker()
        else:
            return True

    def is_collision_free(self, p: Point, q: Quaternion):
        if self.collision_object_name is None:
            raise Exception(u'Please set object for {}.'.format(str(self)))
        if self.pybullet_initialized:
            self.update_object_pose(p, q)
            return self.collision_checker()
        else:
            return True


class ThreeDimStateValidator(ob.StateValidityChecker):

    def __init__(self, si, collision_checker):
        ob.StateValidityChecker.__init__(self, si)
        self.collision_checker = collision_checker
        self.lock = threading.Lock()

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


class CompoundBoxSpace:

    def __init__(self, world, robot, map_frame, collision_checker, min_size, start_and_end_positions,
                 publish_collision_boxes=True):

        self.world = world
        self.robot = robot
        self.map_frame = map_frame
        self.publish_collision_boxes = publish_collision_boxes
        self.collisionChecker = collision_checker
        self.min_size = min_size
        self.start_and_end_positions = start_and_end_positions

        if self.publish_collision_boxes:
            self.pub_collision_marker = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)

    def _create_collision_box(self, pose, pos_a, pos_b, collision_sphere_name):
        dist = np.sqrt(np.sum((np.array(pos_a) - np.array(pos_b))**2))
        world_body_box = make_world_body_box(name=collision_sphere_name,
                                             x_length=dist,
                                             y_length=self.min_size,
                                             z_length=self.min_size)
        self.world.add_world_body(world_body_box, pose)
        world_body_box = make_world_body_box(name=collision_sphere_name+'start',
                                             x_length=self.min_size,
                                             y_length=self.min_size,
                                             z_length=self.min_size)
        self.world.add_world_body(world_body_box, Pose(Point(pos_a[0], pos_a[1], pos_a[2]), Quaternion(0,0,0,1)))
        world_body_box = make_world_body_box(name=collision_sphere_name+'end',
                                             x_length=self.min_size,
                                             y_length=self.min_size,
                                             z_length=self.min_size)
        self.world.add_world_body(world_body_box, Pose(Point(pos_b[0], pos_b[1], pos_b[2]), Quaternion(0,0,0,1)))

    def _get_pitch(self, pos_a, pos_b):
        dx = pos_b[0] - pos_a[0]
        dy = pos_b[1] - pos_a[1]
        dz = pos_b[2] - pos_a[2]
        pos_c = pos_a + np.array([dx, dy, 0])
        a = np.sqrt(np.sum((np.array(pos_c) - np.array(pos_a)) ** 2))
        return -np.arctan2(dz, a)

    def _get_yaw(self, pos_a, pos_b):
        dx = pos_b[0] - pos_a[0]
        dz = pos_b[2] - pos_a[2]
        pos_c = pos_a + np.array([0, 0, dz])
        pos_d = pos_c + np.array([dx, 0, 0])
        g = np.sqrt(np.sum((np.array(pos_b) - np.array(pos_d)) ** 2))
        a = np.sqrt(np.sum((np.array(pos_c) - np.array(pos_d)) ** 2))
        return np.arctan2(g, a)

    def get_collisions(self, collision_box_name_prefix=u'is_object_pickable_box'):
        collisions = dict()
        if self.robot:
            collisions_per_pos = dict()
            # Get em
            for i, (pos_a, pos_b) in enumerate(zip(self.start_and_end_positions[:-1], self.start_and_end_positions[1:])):
                if pos_a == pos_b:
                    continue
                b_to_a = np.array(pos_a) - np.array(pos_b)
                c = np.array(pos_b) + b_to_a / 2.
                # https://i.stack.imgur.com/f190Q.png, https://stackoverflow.com/questions/58469297/how-do-i-calculate-the-yaw-pitch-and-roll-of-a-point-in-3d/58469298#58469298
                q = tf.transformations.quaternion_from_euler(0, self._get_pitch(pos_a, pos_b), self._get_yaw(pos_a, pos_b))
                rospy.logerr(u'pitch: {}, yaw: {}'.format(self._get_pitch(pos_a, pos_b), self._get_yaw(pos_a, pos_b)))
                collision_box_name_i = u'{}_{}'.format(collision_box_name_prefix, str(i))
                #if self.is_tip_object():
                #    tip_coll_aabb = p.getAABB(self.robot.get_pybullet_id(),
                #                              self.robot.get_pybullet_link_id(self.tip_link))
                #    min_size = np.min(np.array(tip_coll_aabb))
                #else:
                #    min_size = box_max_size
                self._create_collision_box(Pose(Point(c[0], c[1], c[2]), Quaternion(q[0], q[1], q[2], q[3]),
                                                pos_a, pos_b, collision_box_name_i))
                #self.world.set_object_pose(collision_box_name_i, Pose(Point(c[0], c[1], c[2]),
                #                                                      Quaternion(q[0], q[1], q[2], q[3])))
                if self.publish_collision_boxes:
                    self.pub_marker(collision_box_name_i)
                self.collisionChecker.update_collisions_environment()
                if not self.collisionChecker.is_object_external_collision_free(collision_box_name_i):
                    collisions_per_pos[i] = self.collisionChecker.get_collisions(collision_box_name_i)
                if self.publish_collision_boxes:
                    self.del_marker(collision_box_name_i)
                self.world.remove_object(collision_box_name_i)
                self.world.remove_object(collision_box_name_i+'start')
                self.world.remove_object(collision_box_name_i+'end')
        return collisions

    def pub_marker(self, name):
        names = [name, name+'start', name+'end']
        ma = MarkerArray()
        for n in names:
            m = self.world.get_object(n).as_marker_msg()
            m.header.frame_id = self.map_frame
            m.ns = u'world' + m.ns
            ma.markers.append(m)
        self.pub_collision_marker.publish(ma)

    def del_marker(self, name):
        names = [name, name+'start', name+'end']
        ma = MarkerArray()
        for n in names:
            m = self.world.get_object(n).as_marker_msg()
            m.action = m.DELETE
            m.ns = u'world' + m.ns
            ma.markers.append(m)
        self.pub_collision_marker.publish(ma)

class GlobalPlanner(GetGoal):

    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)

        #self.robot = self.robot
        self.map_frame = self.get_god_map().get_data(identifier.map_frame)
        self.l_tip = 'l_gripper_tool_frame'
        self.r_tip = 'r_gripper_tool_frame'

        self.kitchen_space = self.create_kitchen_space()
        self.kitchen_floor_space = self.create_kitchen_floor_space()

        self.pose_goal = None
        self.__goal_dict = None

        self._planner_solve_params = {}  # todo: load values from rosparam
        self.navigation_config = 'slow_without_refine'  # todo: load value from rosparam
        self.movement_config = 'slow_without_refine'

        self.initialised_planners = False
        self.collisionCheckerInterface = CollisionCheckerInterface()

    def get_cart_goal(self, cmd):
        try:
            return next(c for c in cmd.constraints if c.type == "CartesianPose")
        except StopIteration:
            return None

    def is_global_path_needed(self, env_group='kitchen'):
        env_link_names = self.world.groups[env_group].link_names_with_collisions
        ids = list()
        for l in env_link_names:
            if l in self.collision_scene.object_name_to_bullet_id:
                ids.append(self.collision_scene.object_name_to_bullet_id[l])
            else:
                raise Exception(u'Link {} with collision was not added in the collision scene.'.format(l))
        return self.__is_global_path_needed(ids)

    def __is_global_path_needed(self, coll_body_ids):
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
        pose_matrix = self.get_robot().get_fk(self.root_link, self.tip_link)
        curr_R_pose = np_to_pose_stamped(pose_matrix, self.root_link)
        curr_pos = transform_pose(self.get_god_map().get_data(identifier.map_frame), curr_R_pose).pose.position
        curr_arr = numpy.array([curr_pos.x, curr_pos.y, curr_pos.z])
        goal_pos = self.pose_goal.pose.position
        goal_arr = numpy.array([goal_pos.x, goal_pos.y, goal_pos.z])
        obj_id, _, _, _, normal = p.rayTest(curr_arr, goal_arr)[0]
        if obj_id in coll_body_ids:
            diff = numpy.sqrt(numpy.sum((curr_arr - goal_arr) ** 2))
            v = numpy.array(list(normal)) * diff
            new_curr_arr = goal_arr + v
            obj_id, _, _, _, _ = p.rayTest(new_curr_arr, goal_arr)[0]
            return obj_id in coll_body_ids
        else:
            return False

    def is_global_navigation_needed(self):
        return self.tip_link == u'base_footprint' and \
               self.root_link == self.get_robot().root_link.name and \
               self.is_global_path_needed()

    def is_tip_object(self):
        return self.robot.get_parent_link_of_link(self.tip_link) in [self.r_tip, self.l_tip]

    def save_cart_goal(self, cart_c):

        self.__goal_dict = yaml.load(cart_c.parameter_value_pair)
        ros_pose = convert_dictionary_to_ros_message(self.__goal_dict[u'goal'])
        self.pose_goal = transform_pose(self.get_god_map().get_data(identifier.map_frame), ros_pose)

        self.root_link = self.__goal_dict[u'root_link']
        self.tip_link = self.__goal_dict[u'tip_link']
        link_names = self.get_robot().link_names

        if self.root_link not in link_names:
            raise Exception(u'Root_link {} is no known link of the robot.'.format(self.root_link))
        if self.tip_link not in link_names:
            raise Exception(u'Tip_link {} is no known link of the robot.'.format(self.tip_link))
        if not self.get_robot().are_linked(self.root_link, self.tip_link):
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

        if not self.initialised_planners:
            self.setupNavigation()
            self.setupMovement()
            self.initialised_planners = True

        # Parse and save the Cartesian Goal Constraint
        self.save_cart_goal(cart_c)

        self.collisionCheckerInterface.update_collisions_environment()
        if self.is_global_navigation_needed():
            trajectory = self.planNavigation()
        elif self.is_global_path_needed():
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
            if len(point) > 3:
                base_pose.pose.orientation = Quaternion(point[3], point[4], point[5], point[6])
            else:
                arr = tf.transformations.quaternion_from_euler(0, 0, point[2])
                base_pose.pose.orientation = Quaternion(arr[0], arr[1], arr[2], arr[3])
            poses.append(base_pose)
        # poses[-1].pose.orientation = self.pose_goal.pose.orientation
        move_cmd.constraints = [self.get_cartesian_path_constraints(poses)]
        self.get_god_map().set_data(identifier.next_move_goal, move_cmd)
        return Status.SUCCESS

    def get_cartesian_path_constraints(self, poses):

        d = dict()
        d[u'parameter_value_pair'] = deepcopy(self.__goal_dict)
        d[u'parameter_value_pair'].pop(u'goal')

        c_d = deepcopy(d)
        c_d[u'parameter_value_pair'][u'goals'] = list(map(convert_ros_message_to_dictionary, poses))
        c = Constraint()
        c.type = u'CartesianPathCarrot'
        c.parameter_value_pair = json.dumps(c_d[u'parameter_value_pair'])

        return c

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

    def get_start_state(self, space, round_decimal_place=3):
        matrix_root_linkTtip_link = self.get_robot().get_fk(self.root_link, self.tip_link)
        pose_root_linkTtip_link = np_to_pose_stamped(matrix_root_linkTtip_link, self.root_link)
        pose_mTtip_link = transform_pose(self.get_god_map().get_data(identifier.map_frame), pose_root_linkTtip_link)
        return self._get_state(space, pose_mTtip_link, round_decimal_place=round_decimal_place)

    def get_goal_state(self, space, round_decimal_place=3):
        return self._get_state(space, self.pose_goal, round_decimal_place=round_decimal_place)

    def _get_state(self, space, pose_stamped, round_decimal_place=3):
        state = ob.State(space)
        state().setX(round(pose_stamped.pose.position.x, round_decimal_place))
        state().setY(round(pose_stamped.pose.position.y, round_decimal_place))
        o = pose_stamped.pose.orientation
        if is_3D(space):
            state().setZ(round(pose_stamped.pose.position.z, round_decimal_place))
            state().rotation().x = o.x
            state().rotation().y = o.y
            state().rotation().z = o.z
            state().rotation().w = o.w
        else:
            angles = tf.transformations.euler_from_quaternion([o.x, o.y, o.z, o.w])
            state().setYaw(angles[2])
        # Force goal and start pose into limited search area
        space.enforceBounds(state())
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
        self._planner_solve_params['kABITstar'] = {
            'slow_without_refine': SolveParameters(initial_solve_time=60, refine_solve_time=5, max_initial_iterations=3,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'slow_with_refine': SolveParameters(initial_solve_time=60, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5),
            'fast_without_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'fast_with_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5)
        }
        self._planner_solve_params['ABITstar'] = {
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
        si.setMotionValidator(RayMotionValidator(si, False, self.get_robot(), self.collision_scene, tip_link=u'base_footprint'))
        collision_checker = RobotBulletCollisionChecker(False, tip_link=u'base_footprint')
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
        si.setMotionValidator(RayMotionValidator(si, True, self.get_robot(), self.collision_scene))
        collision_checker = RobotBulletCollisionChecker(True)
        si.setStateValidityChecker(ThreeDimStateValidator(si, collision_checker))
        si.setup()

        # si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(self.allocOBValidStateSampler))
        # si.setStateValidityCheckingResolution(0.001)

        # Set navigation planner
        planner = self.get_navigation_planner(si)
        self.movement_setup.setPlanner(planner)

    def planNavigation(self, plot=True):

        # Set Start and Goal pose for Planner
        start = self.get_start_state(self.kitchen_floor_space)
        goal = self.get_goal_state(self.kitchen_floor_space)
        self.navigation_setup.setStartAndGoalStates(start, goal)

        # Set Optimization Function
        optimization_objective = PathLengthAndGoalOptimizationObjective(self.navigation_setup.getSpaceInformation(),
                                                                        goal)
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
            data = states_matrix_str2array_floats(
                self.navigation_setup.getSolutionPath().printAsMatrix())  # [[x, y, theta]]
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
        # self.movement_setup.getSpaceInformation().getMotionValidator().update_tip_link(self.tip_link)
        si = self.movement_setup.getSpaceInformation()
        si.setMotionValidator(RayMotionValidator(si, True, self.get_robot(), self.collision_scene, tip_link=self.tip_link))

        start = self.get_start_state(self.kitchen_space)
        goal = self.get_goal_state(self.kitchen_space)
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

            data = states_matrix_str2array_floats(
                self.movement_setup.getSolutionPath().printAsMatrix())  # [x y z xw yw zw w]
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

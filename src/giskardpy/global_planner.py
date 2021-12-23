#!/usr/bin/env python

import json
import sys
import threading
from collections import namedtuple
from time import sleep

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
import giskardpy.model.pybullet_wrapper as pbw

from copy import deepcopy

import giskardpy.identifier as identifier
import giskardpy.model.pybullet_wrapper as pw
from giskardpy.exceptions import GlobalPlanningException
from giskardpy.model.utils import make_world_body_box
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.tree.get_goal import GetGoal
from giskardpy.utils.kdl_parser import KDL
from giskardpy.utils.tfwrapper import transform_pose, lookup_pose, np_to_pose_stamped, list_to_kdl, pose_to_np

from mpl_toolkits.mplot3d import Axes3D
import numpy
import matplotlib.pyplot as plt
from ompl import base as ob
from ompl import geometric as og

# todo: put below in ros params
SolveParameters = namedtuple('SolveParameters', 'initial_solve_time refine_solve_time max_initial_iterations '
                                                'max_refine_iterations min_refine_thresh')
CollisionAABB = namedtuple('CollisionAABB', 'link d u')

from giskardpy.utils.utils import convert_dictionary_to_ros_message, convert_ros_message_to_dictionary


class CollisionCheckerInterface(GiskardBehavior):

    def __init__(self, name='CollisionCheckerInterface'):
        super().__init__(name)

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


class CollisionObjects(object):

    def __init__(self):
        self.collision_objects = list()

    def update(self):
        raise Exception('not implemented')

    def add_collision(self, link_name):
        raise Exception('not implemented')

    def get_collision(self, link_name):
        raise Exception('not implemented')


class AABBCollision(CollisionObjects):

    def __init__(self):
        super().__init__()

    def add_collision(self, link_name):
        link_names = [l for (l, _, _) in self.collision_objects]
        if link_name not in link_names:
            self.collision_objects.append(self.get_collision(link_name))

    def get_points(self, x_b, y_b, z_b, x_e, y_e, z_e, transform_points=None):
        query_b = list()
        query_e = list()
        for collision_info in self.collision_objects:
            aabbs = self.aabb_to_3d_points(collision_info.d, collision_info.u)
            q_b = [[x_b + aabb[0], y_b + aabb[1], z_b + aabb[2]] for aabb in aabbs]
            q_e = [[x_e + aabb[0], y_e + aabb[1], z_e + aabb[2]] for aabb in aabbs]
            # If the aabb box is not from the self.tip_link, add the translation from the
            # self.tip_link to the collision info link.
            if transform_points is not None:
                if collision_info.link in transform_points:
                    point = transform_points[collision_info.link]
                    q_b = [[point[0] + q_b[0], point[1] + q_b[1], point[2] + q_b[2]] for q_b in deepcopy(q_b)]
                    q_e = [[point[0] + q_e[0], point[1] + q_e[1], point[2] + q_e[2]] for q_e in deepcopy(q_e)]
            query_b.extend(q_b)
            query_e.extend(q_e)
        return query_b, query_e

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


class GiskardLinkCollision(object):

    def __init__(self, object_in_motion, tip_link):
        self.object_in_motion = object_in_motion
        self.tip_link = tip_link

    def get_links(self):
        controlled_parent_joint_name = self.object_in_motion.get_controlled_parent_joint_of_link(self.tip_link)
        child_links_with_collision, _ = self.object_in_motion.search_branch(controlled_parent_joint_name,
                                                                            collect_link_when=self.object_in_motion.has_link_collisions)
        return child_links_with_collision


class GiskardPyBulletAABBCollision(AABBCollision, GiskardLinkCollision):

    def __init__(self, object_in_motion, collision_scene, tip_link, map_frame='map'):
        AABBCollision.__init__(self)
        GiskardLinkCollision.__init__(self, object_in_motion, tip_link)
        self.map_frame = map_frame
        self.collision_scene = collision_scene
        self.update()

    def update(self):
        """
        If tip link is not used for interaction and has collision information, we use only this information for
        continuous collision checking. Otherwise, we check the parent links of the tip link for fixed links and
        their collision information. If there were no fixed parent links with collision information added,
        we search for collision information in the childs links if possible.
        Further we model interaction tip_links such that it makes sense to search,
        for neighboring links and their collision information. Since the joints
        of these links are normally not fixed, we assume that they are.

        """
        # Add collisions
        self.collision_objects = list()
        for link_name in self.get_links():
            self.add_collision(link_name)
        # Add transforms towards the collision link wrt the tip_link
        self.transform_points = dict()
        for collision_info in self.collision_objects:
            if collision_info.link != self.tip_link:
                if collision_info.link in self.transform_points:
                    continue
                else:
                    map_2_tip_link_pos = lookup_pose(self.map_frame, self.tip_link).pose.position
                    map_2_coll_link_pos = lookup_pose(self.map_frame, collision_info.link).pose.position
                    point = [map_2_coll_link_pos.x - map_2_tip_link_pos.x,
                             map_2_coll_link_pos.y - map_2_tip_link_pos.y,
                             map_2_coll_link_pos.z - map_2_tip_link_pos.z]
                    self.transform_points[collision_info.link] = point

    def get_points(self, x_b, y_b, z_b, x_e, y_e, z_e, transform_points=None):
        self.update()
        return super().get_points(x_b, y_b, z_b, x_e, y_e, z_e, transform_points=self.transform_points)

    def get_collision(self, link_name):
        if self.object_in_motion.has_link_collisions(link_name):
            link_id = self.collision_scene.object_name_to_bullet_id[link_name]
            cur_pose = self.collision_scene.fks[link_name]
            cur_pos = cur_pose[:3]
            aabbs = p.getAABB(link_id, physicsClientId=0)
            aabbs_ind = [aabb - cur_pos for aabb in aabbs]
            return CollisionAABB(link_name, aabbs_ind[0], aabbs_ind[1])


class AbstractMotionValidator(ob.MotionValidator):

    """
    This class ensures that every Planner in OMPL makes the same assumption for
    planning edges in the resulting path. The real motion checking must be implemented
    by overwriting the function ompl_check_motion.
    """

    def __init__(self, si, is_3D, tip_link):
        ob.MotionValidator.__init__(self, si)
        self.si = si
        self.state_validator = None
        self.lock = threading.Lock()
        self.is_3D = is_3D
        self.tip_link = tip_link

    def checkMotion(self, s1, s2):
        if self.state_validator is None:
            self.state_validator = self.si.getStateValidityChecker()
        res_a = self.ompl_check_motion(s1, s2)
        res_b = self.ompl_check_motion(s2, s1)
        return res_a and res_b and self.state_validator.isValid(s1) and self.state_validator.isValid(s2)

    def ompl_check_motion(self, s1, s2):
        raise Exception('Please overwrite me')


class PyBulletRayTester(object):

    def __init__(self, environment_description='kitchen_description', init_js=None, ignore_objects_ids=None):
        self.once = False
        self.link_id_start = -1
        self.collision_free_id = -1
        self.collisionFilterGroup = 0x1
        self.noCollisionFilterGroup = 0x0
        self.environment_id = None
        self.environment_description = environment_description
        self.init_js = init_js
        if ignore_objects_ids is None:
            self.ignore_object_ids = list()
        else:
            self.ignore_object_ids = ignore_objects_ids
        self.setup()

    def setup(self):
        for i in range(0, 100):
            if not pbw.p.isConnected(physicsClientId=i):
                self.client_id = i
                break
        pbw.start_pybullet(False, client_id=self.client_id)
        self.environment_id = pbw.load_urdf_string_into_bullet(rospy.get_param(self.environment_description),
                                                               client_id=self.client_id)
        self.update(self.init_js)

    def update(self, js):
        if js is not None:
            for joint_id in range(0, pbw.p.getNumJoints(self.environment_id, physicsClientId=self.client_id)):
                joint_name = pbw.p.getJointInfo(self.environment_id, joint_id, physicsClientId=self.client_id)[
                    1].decode()
                joint_state = js[joint_name].position
                pbw.p.resetJointState(self.environment_id, joint_id, joint_state, physicsClientId=self.client_id)
            #pbw.p.stepSimulation(physicsClientId=self.client_id)

    def pre_ray_test(self):
        bodies_num = p.getNumBodies(physicsClientId=self.client_id)
        if bodies_num > 1:
            for id in range(0, bodies_num):
                links_num = p.getNumJoints(id, physicsClientId=self.client_id)
                for link_id in range(self.link_id_start, links_num):
                    if id not in self.ignore_object_ids:
                        p.setCollisionFilterGroupMask(id, link_id, self.collisionFilterGroup, self.collisionFilterGroup,
                                                      physicsClientId=self.client_id)
                    else:
                        p.setCollisionFilterGroupMask(id, link_id, self.noCollisionFilterGroup,
                                                      self.noCollisionFilterGroup, physicsClientId=self.client_id)

    def ray_test_batch(self, js, rayFromPositions, rayToPositions):
        if not self.once:
            self.update(js)
            self.once = True
        bodies_num = p.getNumBodies(physicsClientId=self.client_id)
        if bodies_num > 1:
            query_res = p.rayTestBatch(rayFromPositions, rayToPositions, numThreads=0, physicsClientId=self.client_id,
                                       collisionFilterMask=self.collisionFilterGroup)
        else:
            query_res = p.rayTestBatch(rayFromPositions, rayToPositions, numThreads=0, physicsClientId=self.client_id)
        if any(v[0] in self.ignore_object_ids for v in query_res):
            rospy.logerr('fak')
        coll_links = []
        for i in range(0, len(query_res)):
            obj_id = query_res[i][0]
            if obj_id != self.collision_free_id:
                coll_links.append(p.getBodyInfo(obj_id)[0])
        return all([v[0] == self.collision_free_id for v in query_res]), coll_links

    def post_ray_test(self):
        bodies_num = p.getNumBodies(physicsClientId=self.client_id)
        if bodies_num > 1:
            for id in range(0, p.getNumBodies(physicsClientId=self.client_id)):
                links_num = p.getNumJoints(id, physicsClientId=self.client_id)
                for link_id in range(self.link_id_start, links_num):
                    p.setCollisionFilterGroupMask(id, link_id, self.collisionFilterGroup, self.collisionFilterGroup,
                                                  physicsClientId=self.client_id)

    def close_pybullet(self):
        pbw.stop_pybullet(client_id=self.client_id)


class RayMotionValidator(AbstractMotionValidator):

    def __init__(self, si, is_3D, object_in_motion, collision_scene, tip_link, debug=False, raytester=None, js=None,
                 points_supplier=None):
        AbstractMotionValidator.__init__(self, si, is_3D, tip_link)
        self.hitting = {}
        self.debug = debug
        self.js = js
        if raytester is None:
            self.raytester = PyBulletRayTester()
        else:
            self.raytester = raytester
        if points_supplier is None:
            self.collision_points = GiskardPyBulletAABBCollision(object_in_motion, collision_scene, tip_link)
        else:
            self.collision_points = points_supplier

    def ompl_check_motion(self, s1, s2):
        with self.lock:
            self.raytester.pre_ray_test()
            res = self._ompl_check_motion(s1, s2)
            self.raytester.post_ray_test()
            return res

    def check_motion(self, s1, s2):
        with self.lock:
            self.raytester.pre_ray_test()
            res = self._check_motion(s1[0], s1[1], s1[2], s2[0], s2[1], s2[2])
            self.raytester.post_ray_test()
            return res

    def _check_motion(self, x_b, y_b, z_b, x_e, y_e, z_e):
        # Shoot ray from start to end pose and check if it intersects with the kitchen,
        # if so return false, else true.
        query_b, query_e = self.collision_points.get_points(x_b, y_b, z_b, x_e, y_e, z_e)
        collision_free, coll_links = self.raytester.ray_test_batch(self.js, query_b, query_e)
        return collision_free

    def _ompl_check_motion(self, s1, s2):
        x_b = s1.getX()
        y_b = s1.getY()
        x_e = s2.getX()
        y_e = s2.getY()
        if self.is_3D:
            z_b = s1.getZ()
            z_e = s2.getZ()
        else:
            z_b = 0
            z_e = 0
        return self._check_motion(x_b, y_b, z_b, x_e, y_e, z_e)


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


class IK(object):

    def __init__(self, root_link, tip_link):
        self.root_link = root_link
        self.tip_link = tip_link

    def setup(self):
        pass

    def get_ik(self, old_js, pose):
        pass


class KDLIK(IK):

    def __init__(self, root_link, tip_link, static_joints=None, robot_description='robot_description'):
        IK.__init__(self, root_link, tip_link)
        self.robot_description = rospy.get_param(robot_description)
        self._kdl_robot = None
        self.robot_kdl_tree = None
        self.static_joints = static_joints
        self.setup()

    def setup(self):
        self.robot_kdl_tree = KDL(self.robot_description)
        self._kdl_robot = self.robot_kdl_tree.get_robot(self.root_link, self.tip_link, static_joints=self.static_joints)

    def get_ik(self, js, pose):
        new_js = deepcopy(js)
        js_dict_position = {}
        for k, v in js.items():
            js_dict_position[k] = v.position
        joint_array = self._kdl_robot.ik(js_dict_position, list_to_kdl(pose))
        for i, joint_name in enumerate(self._kdl_robot.joints):
            new_js[joint_name].position = joint_array[i]
        return new_js


class PyBulletIK(IK):

    def __init__(self, root_link, tip_link, static_joints=None, robot_description='robot_description'):
        IK.__init__(self, root_link, tip_link)
        self.robot_description = rospy.get_param(robot_description)
        self.pybullet_joints = list()
        self.robot_id = None
        self.pybullet_tip_link_id = None
        self.once = False
        self.static_joints = static_joints
        self.joint_lowers = list()
        self.joint_uppers = list()
        self.setup()

    def setup(self):
        for i in range(0, 100):
            if not pbw.p.isConnected(physicsClientId=i):
                self.client_id = i
                break
        pbw.start_pybullet(False, client_id=self.client_id)
        pos, q = pose_to_np(lookup_pose('map', self.root_link).pose)
        self.robot_id = pbw.load_urdf_string_into_bullet(self.robot_description, position=pos,
                                                         orientation=q, client_id=self.client_id)
        for i in range(0, pbw.p.getNumJoints(self.robot_id, physicsClientId=1)):
            j = pbw.p.getJointInfo(self.robot_id, i, physicsClientId=1)
            if j[2] != p.JOINT_FIXED:
                joint_name = j[1].decode('UTF-8')
                if self.static_joints and joint_name not in self.static_joints:
                    self.joint_lowers.append(j[8])
                    self.joint_uppers.append(j[9])
                else:
                    self.joint_lowers.append(0)
                    self.joint_uppers.append(0)
                self.pybullet_joints.append(joint_name)
            if j[12].decode('UTF-8') == self.tip_link:
                self.pybullet_tip_link_id = j[0]

    def get_ik(self, js, pose):
        if not self.once:
            self.update_pybullet(js)
            self.once = True
        new_js = deepcopy(js)
        state_ik = p.calculateInverseKinematics(self.robot_id, self.pybullet_tip_link_id,
                                                pose[0], pose[1], self.joint_lowers, self.joint_uppers,
                                                physicsClientId=self.client_id)
        for joint_name, joint_state in zip(self.pybullet_joints, state_ik):
            new_js[joint_name].position = joint_state
        return new_js

    def update_pybullet(self, js):
        for joint_id in range(0, pbw.p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            joint_name = pbw.p.getJointInfo(self.robot_id, joint_id, physicsClientId=self.client_id)[1].decode()
            joint_state = js[joint_name].position
            pbw.p.resetJointState(self.robot_id, joint_id, joint_state, physicsClientId=self.client_id)
        #pbw.p.stepSimulation(physicsClientId=self.client_id)

    def close_pybullet(self):
        pbw.stop_pybullet(client_id=self.client_id)


class RobotBulletCollisionChecker(GiskardBehavior):

    def __init__(self, is_3D, root_link, tip_link, name='RobotBulletCollisionChecker', ik=None):
        super().__init__(name)
        self.pybullet_lock = threading.Lock()
        if ik is None:
            self.ik = PyBulletIK(root_link, tip_link)
        else:
            self.ik = ik(root_link, tip_link)
        self.is_3D = is_3D
        self.tip_link = tip_link
        self.setup_pybullet()

    def setup_pybullet(self):
        for i in range(0, 100):
            if not pbw.p.isConnected(physicsClientId=i):
                self.client_id = i
                break
        pbw.start_pybullet(False, client_id=self.client_id)
        self.robot_id = pbw.load_urdf_string_into_bullet(rospy.get_param('robot_description'), client_id=self.client_id)
        self.kitchen_id = pbw.load_urdf_string_into_bullet(rospy.get_param('kitchen_description'),
                                                           client_id=self.client_id)
        self.pybullet_joints_id = dict()
        for i in range(0, p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            j = p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            if j[2] != p.JOINT_FIXED:
                self.pybullet_joints_id[j[1].decode('UTF-8')] = i

    def is_collision_free(self, x, y, z, rot):
        with self.pybullet_lock:
            # Get current joint states
            current_js = self.get_god_map().get_data(identifier.joint_states)
            # Calc IK for navigating to given state and ...
            if self.is_3D:
                pose = [[x, y, z], list(rot)]
            else:
                pose = [[x, y, 0], list(rot)]
            results = []
            state_ik = self.ik.get_ik(current_js, pose)
            # override on current joint states.
            for joint_name, id in self.pybullet_joints_id.items():
                if joint_name not in ['odom_x_joint', 'odom_y_joint', 'odom_z_joint']:
                    pbw.p.resetJointState(self.robot_id, id, state_ik[joint_name].position,
                                          physicsClientId=self.client_id)
            pbw.p.resetBasePositionAndOrientation(self.robot_id,
                                                  [state_ik['odom_x_joint'].position,
                                                   state_ik['odom_y_joint'].position,
                                                   0],
                                                  p.getQuaternionFromEuler([0, 0, state_ik['odom_z_joint'].position]),
                                                  physicsClientId=self.client_id)
            #pbw.p.stepSimulation(physicsClientId=self.client_id)
            # Check if kitchen is colliding with robot
            for i in range(0, pbw.p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
                aabb = pbw.p.getAABB(self.robot_id, i, physicsClientId=self.client_id)
                aabb = [[v - 0.1 for v in aabb[0]], [v + 0.1 for v in aabb[0]]]
                objs = pbw.p.getOverlappingObjects(aabb[0], aabb[1], physicsClientId=self.client_id)
                results.extend(list(filter(lambda o: o[0] == self.kitchen_id, objs)))
            return len(results) == 0

    def ompl_state_to_python(self, state):
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
        return x, y, z, rot

    def is_collision_free_ompl(self, state):
        if self.tip_link is None:
            raise Exception(u'Please set tip_link for {}.'.format(str(self)))
        x, y, z, rot = self.ompl_state_to_python(state)
        return self.is_collision_free(x, y, z, rot)


class GiskardRobotBulletCollisionChecker(GiskardBehavior):

    def __init__(self, is_3D, root_link, tip_link, ik=None, ik_sampling=1, collision_link_names=None):
        GiskardBehavior.__init__(self, str(self))
        self.giskard_lock = threading.Lock()
        if ik is None:
            self.ik = PyBulletIK(root_link, tip_link)
        else:
            self.ik = ik(root_link, tip_link)
        self.is_3D = is_3D
        self.tip_link = tip_link
        self.ik_sampling = ik_sampling
        if collision_link_names is None:
            self.collision_link_names = GiskardLinkCollision(self.robot, tip_link).get_links()
        else:
            self.collision_link_names = collision_link_names

    def is_collision_free(self, x, y, z, rot):
        if True: #with self.giskard_lock:
            # Get current joint states
            old_js = self.get_god_map().get_data(identifier.joint_states)
            # Calc IK for navigating to given state and ...
            if self.is_3D:
                pose = [[x, y, z], rot]
            else:
                pose = [[x, y, 0], rot]
            results = []
            for i in range(0, self.ik_sampling):
                state_ik = self.ik.get_ik(old_js, pose)
                # override on current joint states.
                self.robot.state = state_ik
                # Check if kitchen is colliding with robot
                results.append(self.collision_scene.are_robot_links_external_collision_free(self.collision_link_names, dist=0))
                # Reset joint state
                self.get_robot().state = old_js
        return any(results)

    def get_furthest_normal(self, x, y, z, rot):
        # Get current joint states
        old_js = self.get_god_map().get_data(identifier.joint_states)
        # Calc IK for navigating to given state and ...
        if self.is_3D:
            pose = [[x, y, z], rot]
        else:
            pose = [[x, y, 0], rot]
        state_ik = self.ik.get_ik(old_js, pose)
        # override on current joint states.
        self.get_robot().state = state_ik
        # Check if kitchen is colliding with robot
        result = self.collision_scene.get_furthest_normal(self.collision_link_names)
        # Reset joint state
        self.get_robot().state = old_js
        return result

    def ompl_state_to_python(self, state):
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
        return x, y, z, list(rot)

    def is_collision_free_ompl(self, state):
        if self.tip_link is None:
            raise Exception(u'Please set tip_link for {}.'.format(str(self)))
        x, y, z, rot = self.ompl_state_to_python(state)
        return self.is_collision_free(x, y, z, rot)

    def get_furthest_normal_ompl(self, state):
        if self.tip_link is None:
            raise Exception(u'Please set tip_link for {}.'.format(str(self)))
        x, y, z, rot = self.ompl_state_to_python(state)
        return self.get_furthest_normal(x, y, z, rot)


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

    def isValid(self, state):
        with self.lock:
            return self.collision_checker.is_collision_free_ompl(state)

    def __str__(self):
        return u'ThreeDimStateValidator'


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
            return self.collision_checker.is_collision_free_ompl(state) and self.is_driveable(state)

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
        dist = np.sqrt(np.sum((np.array(pos_a) - np.array(pos_b)) ** 2))
        world_body_box = make_world_body_box(name=collision_sphere_name,
                                             x_length=dist,
                                             y_length=self.min_size,
                                             z_length=self.min_size)
        self.world.add_world_body(world_body_box, pose)
        world_body_box = make_world_body_box(name=collision_sphere_name + 'start',
                                             x_length=self.min_size,
                                             y_length=self.min_size,
                                             z_length=self.min_size)
        self.world.add_world_body(world_body_box, Pose(Point(pos_a[0], pos_a[1], pos_a[2]), Quaternion(0, 0, 0, 1)))
        world_body_box = make_world_body_box(name=collision_sphere_name + 'end',
                                             x_length=self.min_size,
                                             y_length=self.min_size,
                                             z_length=self.min_size)
        self.world.add_world_body(world_body_box, Pose(Point(pos_b[0], pos_b[1], pos_b[2]), Quaternion(0, 0, 0, 1)))

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
            for i, (pos_a, pos_b) in enumerate(
                    zip(self.start_and_end_positions[:-1], self.start_and_end_positions[1:])):
                if pos_a == pos_b:
                    continue
                b_to_a = np.array(pos_a) - np.array(pos_b)
                c = np.array(pos_b) + b_to_a / 2.
                # https://i.stack.imgur.com/f190Q.png, https://stackoverflow.com/questions/58469297/how-do-i-calculate-the-yaw-pitch-and-roll-of-a-point-in-3d/58469298#58469298
                q = tf.transformations.quaternion_from_euler(0, self._get_pitch(pos_a, pos_b),
                                                             self._get_yaw(pos_a, pos_b))
                rospy.logerr(u'pitch: {}, yaw: {}'.format(self._get_pitch(pos_a, pos_b), self._get_yaw(pos_a, pos_b)))
                collision_box_name_i = u'{}_{}'.format(collision_box_name_prefix, str(i))
                # if self.is_tip_object():
                #    tip_coll_aabb = p.getAABB(self.robot.get_pybullet_id(),
                #                              self.robot.get_pybullet_link_id(self.tip_link))
                #    min_size = np.min(np.array(tip_coll_aabb))
                # else:
                #    min_size = box_max_size
                self._create_collision_box(Pose(Point(c[0], c[1], c[2]), Quaternion(q[0], q[1], q[2], q[3]),
                                                pos_a, pos_b, collision_box_name_i))
                # self.world.set_object_pose(collision_box_name_i, Pose(Point(c[0], c[1], c[2]),
                #                                                      Quaternion(q[0], q[1], q[2], q[3])))
                if self.publish_collision_boxes:
                    self.pub_marker(collision_box_name_i)
                self.collisionChecker.update_collisions_environment()
                if not self.collisionChecker.is_object_external_collision_free(collision_box_name_i):
                    collisions_per_pos[i] = self.collisionChecker.get_collisions(collision_box_name_i)
                if self.publish_collision_boxes:
                    self.del_marker(collision_box_name_i)
                self.world.remove_object(collision_box_name_i)
                self.world.remove_object(collision_box_name_i + 'start')
                self.world.remove_object(collision_box_name_i + 'end')
        return collisions

    def pub_marker(self, name):
        names = [name, name + 'start', name + 'end']
        ma = MarkerArray()
        for n in names:
            m = self.world.get_object(n).as_marker_msg()
            m.header.frame_id = self.map_frame
            m.ns = u'world' + m.ns
            ma.markers.append(m)
        self.pub_collision_marker.publish(ma)

    def del_marker(self, name):
        names = [name, name + 'start', name + 'end']
        ma = MarkerArray()
        for n in names:
            m = self.world.get_object(n).as_marker_msg()
            m.action = m.DELETE
            m.ns = u'world' + m.ns
            ma.markers.append(m)
        self.pub_collision_marker.publish(ma)


def verify_ompl_solution(setup, debug=False):
    rbc = setup.getStateValidityChecker().collision_checker
    si = setup.getSpaceInformation()
    t = 0
    f = 0
    tj = states_matrix_str2array_floats(setup.getSolutionPath().printAsMatrix())
    for i in range(0, len(tj)):
        bool = rbc.is_collision_free(tj[i][0], tj[i][1], 0, p.getQuaternionFromEuler([0, 0, tj[i][2]]))
        if bool:
            t += 1
        else:
            f += 1
            log_a = u''
            log_b = u''
            c_a = True
            c_b = True
            s_c = ob.State(si.getStateSpace())
            s_c().setX(tj[i][0])
            s_c().setY(tj[i][1])
            s_c().setYaw(tj[i][2])
            if i > 0:
                s_b = ob.State(si.getStateSpace())
                s_b().setX(tj[i - 1][0])
                s_b().setY(tj[i - 1][1])
                s_b().setYaw(tj[i - 1][2])
                c_b_1 = si.checkMotion(s_b(), s_c())
                c_b_2 = si.checkMotion(s_c(), s_b())
                c_b = c_b_1 and c_b_2
                if c_b_1 != c_b_2:
                    rospy.logerr('before to current is not same as current to before:')
                if not c_b:
                    log_b += u' point_b: {}, point_c: {}'.format(tj[i - 1], tj[i])
            if i != len(tj) - 1:
                s_a = ob.State(si.getStateSpace())
                s_a().setX(tj[i + 1][0])
                s_a().setY(tj[i + 1][1])
                s_a().setYaw(tj[i + 1][2])
                c_a_1 = si.checkMotion(s_c(), s_a())
                c_a_2 = si.checkMotion(s_a(), s_c())
                c_a = c_a_1 and c_a_2
                if c_a_1 != c_a_2:
                    rospy.logerr('current to after is not the same as after to current:')
                if not c_a:
                    log_a += ' point_a: {}'.format(tj[i + 1])
            if not (c_a and c_b):
                log = u'c_b: {}, c_a: {}'.format(c_b, c_a) + log_b + log_a
                rospy.logerr(log)
    if debug:
        rospy.loginfo(u'Num Invalid States: {}, Num Valid States: {}, Rate FP: {}'.format(t, f, f / t))
    return f / t


def allocGiskardValidStateSample(si):
    return GiskardValidStateSample(si)  # ob.GaussianValidStateSampler(si)


class GiskardValidStateSample(ob.ValidStateSampler):
    def __init__(self, si):
        super(GiskardValidStateSample, self).__init__(si)
        self.name_ = "GiskardValidStateSample"
        self.space = si.getStateSpace()
        self.rbc = GiskardRobotBulletCollisionChecker(False, 'odom_combined', 'base_footprint')

    # Generate a sample in the valid part of the R^3 state space.
    # Valid states satisfy the following constraints:
    # -1<= x,y,z <=1
    # if .25 <= z <= .5, then |x|>.8 and |y|>.8
    def sample(self, state):
        r = ob.State(self.space)
        r.random()
        n = self.rbc.get_furthest_normal_ompl(state)
        state.setX(r().getX() + n[0])
        state.setY(r().getY() + n[1])
        if self.rbc.is_3D:
            state().setZ(r().getZ() + n[2])
        if self.rbc.is_collision_free_ompl(state):
            return True
        else:
            return self.sample(state)


class GlobalPlanner(GetGoal):

    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)

        self.supported_cart_goals = ['CartesianPose', 'CartesianPosition', 'CartesianPathCarrot']

        # self.robot = self.robot
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
        self.once = False

    def get_cart_goal(self, cmd):
        try:
            return next(c for c in cmd.constraints if c.type in self.supported_cart_goals)
        except StopIteration:
            return None

    def is_global_navigation_needed(self):
        return self.tip_link == u'base_footprint' and \
               self.root_link == self.get_robot().root_link.name

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

        global_planner_needed = self.god_map.get_data(identifier.global_planner_needed)
        if not global_planner_needed:
            return Status.SUCCESS

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

        if global_planner_needed and self.is_global_navigation_needed():
            self.collision_scene.update_collision_environment()
            map_frame = self.get_god_map().get_data(identifier.map_frame)
            planner = NavigationPlanner(self.kitchen_floor_space, self.collision_scene,
                                        self.robot, self.root_link, self.tip_link, self.pose_goal, map_frame,
                                        config=self.navigation_config)
            js = self.get_god_map().get_data(identifier.joint_states)
            try:
                trajectory = planner.plan(js)
            except GlobalPlanningException:
                self.raise_to_blackboard(GlobalPlanningException())
                return Status.FAILURE
        elif global_planner_needed:
            self.collision_scene.update_collision_environment()
            map_frame = self.get_god_map().get_data(identifier.map_frame)
            planner = MovementPlanner(self.kitchen_space, self.collision_scene,
                                      self.robot, self.root_link, self.tip_link, self.pose_goal, map_frame,
                                      config=self.movement_config)
            js = self.get_god_map().get_data(identifier.joint_states)
            try:
                trajectory = planner.plan(js, once=self.once)
            except GlobalPlanningException:
                self.once = True
                self.raise_to_blackboard(GlobalPlanningException())
                return Status.FAILURE
        else:
            raise GlobalPlanningException('Global planner was called although it is not needed.')

        if len(trajectory) == 0:
            raise GlobalPlanningException('Global Planner did not found a solution.')
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
        self.get_god_map().set_data(identifier.global_planner_needed, False)
        return Status.SUCCESS

    def get_cartesian_path_constraints(self, poses):

        d = dict()
        d[u'parameter_value_pair'] = deepcopy(self.__goal_dict)

        c_d = deepcopy(d)
        c_d[u'parameter_value_pair'][u'goal'] = convert_ros_message_to_dictionary(poses[-1])
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
        bounds.setLow(-10)
        bounds.setHigh(10)
        bounds.setLow(2, 0)
        bounds.setHigh(2, 2)
        space.setBounds(bounds)

        return space

    def create_kitchen_floor_space(self):
        # create an SE2 state space
        space = ob.SE2StateSpace()
        # space.setLongestValidSegmentFraction(0.02)

        # set lower and upper bounds
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(-10)
        bounds.setHigh(10)
        space.setBounds(bounds)

        return space


class OMPLPlanner(object):

    def __init__(self, space, collision_scene, robot, root_link, tip_link, pose_goal, map_frame, config):
        self.setup = None
        self.space = space
        self.collision_scene = collision_scene
        self.robot = robot
        self.root_link = root_link
        self.tip_link = tip_link
        self.pose_goal = pose_goal
        self.map_frame = map_frame
        self.config = config
        self._planner_solve_params = dict()
        self._planner_solve_params['kABITstar'] = {
            'slow_without_refine': SolveParameters(initial_solve_time=30, refine_solve_time=5, max_initial_iterations=3,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'slow_with_refine': SolveParameters(initial_solve_time=60, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5),
            'fast_without_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'fast_with_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5)
        }
        self._planner_solve_params['ABITstar'] = {
            'slow_without_refine': SolveParameters(initial_solve_time=30, refine_solve_time=5, max_initial_iterations=3,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'slow_with_refine': SolveParameters(initial_solve_time=60, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5),
            'fast_without_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'fast_with_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5)
        }
        self._planner_solve_params['RRTConnect'] = {
            'slow_without_refine': SolveParameters(initial_solve_time=120, refine_solve_time=5,
                                                   max_initial_iterations=1,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'slow_with_refine': SolveParameters(initial_solve_time=60, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5),
            'fast_without_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'fast_with_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5)
        }
        self.init_setup()

    def get_planner(self, si):
        raise Exception('Not overwritten.')

    def init_setup(self):

        # create a simple setup object
        self.setup = og.SimpleSetup(self.space)

        # Set two dimensional motion and state validator
        si = self.setup.getSpaceInformation()
        # si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(self.allocOBValidStateSampler))
        # si.setStateValidityCheckingResolution(0.001)

        # Set navigation planner
        planner = self.get_planner(si)
        self.setup.setPlanner(planner)

    def get_start_state(self, space, round_decimal_place=3):
        matrix_root_linkTtip_link = self.robot.get_fk(self.root_link, self.tip_link)
        pose_root_linkTtip_link = np_to_pose_stamped(matrix_root_linkTtip_link, self.root_link)
        pose_mTtip_link = transform_pose(self.map_frame, pose_root_linkTtip_link)
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

    def solve(self):
        # Get solve parameters
        solve_params = self._planner_solve_params[self.setup.getPlanner().getName()][self.config]
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
                planner_status.getStatus() not in [ob.PlannerStatus.EXACT_SOLUTION]:
            planner_status = self.setup.solve(initial_solve_time)
            time_solving_intial += self.setup.getLastPlanComputationTime()
            num_try += 1
        # Refine solution
        refine_iteration = 0
        v_min = 1e6
        time_solving_refine = 0
        if planner_status.getStatus() in [ob.PlannerStatus.EXACT_SOLUTION]:
            while v_min > min_refine_thresh and refine_iteration < max_refine_iterations and \
                    time_solving_refine < max_refine_solve_time:
                if 'ABITstar' in self.setup.getPlanner().getName() and min_refine_thresh is not None:
                    v_before = self.setup.getPlanner().bestCost().value()
                self.setup.solve(refine_solve_time)
                time_solving_refine += self.setup.getLastPlanComputationTime()
                if 'ABITstar' in self.setup.getPlanner().getName() and min_refine_thresh is not None:
                    v_after = self.setup.getPlanner().bestCost().value()
                    v_min = v_before - v_after
                refine_iteration += 1
        return planner_status.getStatus()

class MovementPlanner(OMPLPlanner):

    def __init__(self, kitchen_space, collision_scene, robot, root_link, tip_link, pose_goal, map_frame,
                 config='fast_without_refine'):
        super(MovementPlanner, self).__init__(kitchen_space, collision_scene, robot, root_link, tip_link, pose_goal,
                                              map_frame, config)

    def get_planner(self, si):
        planner = og.RRTConnect(si)
        planner.setRange(0.1)
        # planner.setIntermediateStates(True)
        # planner.setup()
        return planner

    def plan(self, js, plot=True, once=None):

        si = self.setup.getSpaceInformation()
        collision_checker = GiskardRobotBulletCollisionChecker(True, self.root_link, self.tip_link)
        si.setStateValidityChecker(ThreeDimStateValidator(si, collision_checker))
        si.setMotionValidator(RayMotionValidator(si, True, self.robot, self.collision_scene, tip_link=self.tip_link, js=js))
        si.setup()

        start = self.get_start_state(self.space)
        goal = self.get_goal_state(self.space)

        if not once or not si.isValid(start()):
            raise GlobalPlanningException()

        self.setup.setStartAndGoalStates(start, goal)

        optimization_objective = PathLengthAndGoalOptimizationObjective(self.setup.getSpaceInformation(), goal)
        self.setup.setOptimizationObjective(optimization_objective)

        planner_status = self.solve()
        # og.PathSimplifier(si).smoothBSpline(ss.getSolutionPath()) # takes around 20-30 secs with RTTConnect(0.1)
        # og.PathSimplifier(si).reduceVertices(ss.getSolutionPath())
        data = numpy.array([])
        if planner_status in [ob.PlannerStatus.APPROXIMATE_SOLUTION, ob.PlannerStatus.EXACT_SOLUTION]:
            # try to shorten the path
            # self.movement_setup.simplifySolution() DONT! NO! DONT UNCOMMENT THAT! NO! STOP IT! FIRST IMPLEMENT CHECKMOTION! THEN TRY AGAIN!
            # Make sure enough subpaths are available for Path Following
            self.setup.getSolutionPath().interpolate(20)

            data = states_matrix_str2array_floats(
                self.setup.getSolutionPath().printAsMatrix())  # [x y z xw yw zw w]
            # print the simplified path
            if plot:
                plt.close()
                fig, ax = plt.subplots()
                ax.plot(data[:, 1], data[:, 0])
                ax.invert_xaxis()
                ax.set(xlabel='y (m)', ylabel='x (m)',
                       title=u'2D Path in map')
                # ax = fig.gca(projection='2d')
                # ax.plot(data[:, 0], data[:, 1], '.-')
                plt.show()
        self.setup.clear()
        return data


class NavigationPlanner(OMPLPlanner):

    def __init__(self, kitchen_floor_space, collision_scene, robot, root_link, tip_link, pose_goal, map_frame,
                 config='fast_without_refine'):
        super(NavigationPlanner, self).__init__(kitchen_floor_space, collision_scene, robot, root_link, tip_link, pose_goal,
                                                map_frame, config)

    def get_planner(self, si):
        # Navigation:
        # RRTstar, RRTsharp: very sparse(= path length eq max range), high number of fp, slow relative to RRTConnect
        # RRTConnect: many points, low number of fp
        # PRMstar: no set_range function, finds no solutions
        # ABITstar: slow, sparse, but good
        # if self.fast_navigation:
        #    planner = og.RRTConnect(si)
        #    planner.setRange(0.1)
        # else:
        planner = og.RRTConnect(si)
        planner.setRange(0.1)
        # planner.setIntermediateStates(True)
        # planner.setup()
        return planner

    def plan(self, js, plot=True):

        si = self.setup.getSpaceInformation()
        collision_checker = GiskardRobotBulletCollisionChecker(False, self.robot.root_link, u'base_footprint')
        si.setStateValidityChecker(TwoDimStateValidator(si, collision_checker))
        si.setMotionValidator(RayMotionValidator(si, False, self.robot, self.collision_scene, tip_link=u'base_footprint', js=js))
        si.setup()
        # si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(allocGiskardValidStateSample))

        # Set Start and Goal pose for Planner
        start = self.get_start_state(self.space)
        goal = self.get_goal_state(self.space)
        self.setup.setStartAndGoalStates(start, goal)

        # Set Optimization Function
        # optimization_objective = PathLengthAndGoalOptimizationObjective(self.setup.getSpaceInformation(),
        #                                                                goal)
        # self.navigation_setup.setOptimizationObjective(optimization_objective)

        planner_status = self.solve()
        # og.PathSimplifier(si).smoothBSpline(ss.getSolutionPath()) # takes around 20-30 secs with RTTConnect(0.1)
        # og.PathSimplifier(si).reduceVertices(ss.getSolutionPath())
        data = numpy.array([])
        if planner_status in [ob.PlannerStatus.EXACT_SOLUTION]:
            # try to shorten the path
            # ss.simplifySolution() DONT! NO! DONT UNCOMMENT THAT! NO! STOP IT! FIRST IMPLEMENT CHECKMOTION! THEN TRY AGAIN!
            # Make sure enough subpaths are available for Path Following
            self.setup.getSolutionPath().interpolate(20)
            data = states_matrix_str2array_floats(
                self.setup.getSolutionPath().printAsMatrix())  # [[x, y, theta]]
            # print the simplified path
            if plot:
                plt.close()
                fig, ax = plt.subplots()
                ax.plot(data[:, 1], data[:, 0])
                ax.invert_xaxis()
                ax.set(xlabel='y (m)', ylabel='x (m)',
                       title=u'Navigation Path in map - FP: {}'.format(verify_ompl_solution(self.setup, debug=True)))
                # ax = fig.gca(projection='2d')
                # ax.plot(data[:, 0], data[:, 1], '.-')
                plt.show()
        self.setup.clear()
        return data


def is_3D(space):
    return type(space) == type(ob.SE3StateSpace())


def states_matrix_str2array_floats(str: str, line_sep='\n', float_sep=' '):
    states_strings = str.split(line_sep)
    while '' in states_strings:
        states_strings.remove('')
    return numpy.array(list(map(lambda x: numpy.fromstring(x, dtype=float, sep=float_sep), states_strings)))

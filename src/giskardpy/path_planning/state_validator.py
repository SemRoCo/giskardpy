import threading
import time
from copy import deepcopy

import numpy as np
import rospy
import urdf_parser_py.urdf as up
from nav_msgs.srv import GetMap
from pybullet import getAxisAngleFromQuaternion

from giskardpy import identifier
from giskardpy.model.joints import OmniDrive
from giskardpy.path_planning.ik import PyBulletIK
from giskardpy.tree.behaviors.visualization import VisualizationBehavior


def current_milli_time():
    return round(time.time() * 1000)


def get_robot_joint_state(collision_scene):
    state = deepcopy(collision_scene.robot.state)
    omni_joint = collision_scene.world.joints['brumbrum']
    state[omni_joint.x_name] = collision_scene.world.state[omni_joint.x_name]
    state[omni_joint.y_name] = collision_scene.world.state[omni_joint.y_name]
    state[omni_joint.rot_name] = collision_scene.world.state[omni_joint.rot_name]
    return state


def update_joint_state(js, new_js):
    #if any(map(lambda e: type(e) != PrefixName, new_js)):
    #    raise Exception('oi, there are no PrefixNames in yer new_js >:(!')
    js.update((k, new_js[k]) for k in js.keys() and new_js.keys())


def update_robot_state(collision_scene, state):
    update_joint_state(collision_scene.world.state, state)
    collision_scene.world.notify_state_change()
    collision_scene.sync()


def update_robot_pose(collision_scene, pose):
    joint: OmniDrive = collision_scene.world.joints['brumbrum']
    collision_scene.world.state[joint.x_name].position = pose.position.x
    collision_scene.world.state[joint.y_name].position = pose.position.y
    axis, angle = getAxisAngleFromQuaternion([pose.orientation.x,
                                              pose.orientation.y,
                                              pose.orientation.z,
                                              pose.orientation.w])
    if axis[-1] < 0:
        angle = -angle
    collision_scene.world.state[joint.rot_name].position = angle
    collision_scene.world.notify_state_change()
    collision_scene.sync()


class AbstractStateValidator:

    def __init__(self, is_3D):
        self.is_3D = is_3D

    def isValid(self, pose):
        ret = self.is_collision_free(pose)
        if self.is_3D:
            return ret
        else:
            return ret and self.is_driveable(pose)

    def is_collision_free(self, pose):
        raise Exception('Implement me.')

    def is_driveable(self, pose):
        raise Exception('Implement me.')

    def clear(self):
        pass


def fix_remove_transmissions_from_xml_str(robot_description: str):
    s = '<transmission '
    e = '</transmission>'
    final_description = deepcopy(robot_description)
    while s in final_description and e in final_description:
        to_rmv = final_description[final_description.index(s):final_description.index(e)+len(e)]
        final_description = final_description.replace(to_rmv, '')
    return final_description


def omni_drive_to_joints(collision_scene, robot):
    odom_start_name = 'odom_combined'
    robot_start = 'base_footprint'
    link_suffix = '_link'
    zero_origin = up.Pose([0, 0, 0], [0, 0, 0])
    omni_joint = collision_scene.world.joints['brumbrum']
    odom_start = up.Link(name=odom_start_name)
    odom_x = up.Link(name=omni_joint.x_name + link_suffix)
    odom_y = up.Link(name=omni_joint.y_name + link_suffix)
    odom_x_joint = up.Joint(name=omni_joint.x_name,
                            parent=odom_start_name, child=omni_joint.x_name + link_suffix,
                            joint_type='prismatic', axis=[1, 0, 0],
                            limit=up.JointLimit(effort=200.0, velocity=0.5, lower=-1000, upper=1000),
                            origin=deepcopy(zero_origin))
    odom_y_joint = up.Joint(name=omni_joint.y_name,
                            parent=omni_joint.x_name + link_suffix, child=omni_joint.y_name + link_suffix,
                            joint_type='prismatic', axis=[0, 1, 0],
                            limit=up.JointLimit(effort=200.0, velocity=0.5, lower=-1000, upper=1000),
                            origin=deepcopy(zero_origin))
    odom_rot_joint = up.Joint(name=omni_joint.rot_name,
                              parent=omni_joint.y_name + link_suffix, child=robot_start,
                              joint_type='continuous', axis=[0, 0, 1],
                              limit=up.JointLimit(effort=200.0, velocity=0.4),
                              origin=deepcopy(zero_origin))
    robot.add_link(odom_start)
    robot.add_link(odom_x)
    robot.add_link(odom_y)
    robot.add_joint(odom_x_joint)
    robot.add_joint(odom_y_joint)
    robot.add_joint(odom_rot_joint)
    return robot


class GiskardRobotBulletCollisionChecker(AbstractStateValidator):

    def __init__(self, is_3D, root_link, tip_link, collision_scene, god_map,
                 ik=None, ik_sampling=1, publish=True, dist=0.0):
        super().__init__(is_3D)
        self.giskard_lock = threading.Lock()
        if ik is None:
            robot = up.URDF.from_xml_string(fix_remove_transmissions_from_xml_str(rospy.get_param('robot_description')))
            robot_with_base = omni_drive_to_joints(collision_scene, robot)
            self.ik = PyBulletIK(robot_with_base.to_xml_string(), root_link, tip_link)
        else:
            self.ik = ik(root_link, tip_link)
        self.debug = False
        self.debug_times_cc = list()
        self.debug_times = list()
        self.tip_link = tip_link
        self.collision_scene = collision_scene
        self.god_map = god_map
        self.dist = dist
        self.ik_sampling = ik_sampling
        self.collision_link_names = collision_scene.world.get_children_with_collisions_from_link(self.tip_link)
        self.publisher = None
        self.init_map()
        if publish:
            self.publisher = VisualizationBehavior('motion planning object publisher', ensure_publish=False)
            self.publisher.setup(10)

    def get_ik(self, js, pose):
        if self.is_3D:
            return self.ik.get_ik(js, pose)
        else:
            return pose

    def set_tip_link(self, old_js, pose):
        if self.is_3D:
            state = self.ik.get_ik(old_js, pose)
            update_robot_state(self.collision_scene, state)
        else:
            update_robot_pose(self.collision_scene, pose)

    def clear(self):
        self.ik.clear()

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
            tmp = np.zeros((info.height, info.width))
            for x_i in range(0, info.height):
                for y_i in range(0, info.width):
                    tmp[x_i][y_i] = map.data[y_i + x_i * info.width]
            self.occ_map = np.fliplr(deepcopy(tmp))
            self.occ_map_res = info.resolution
            self.occ_map_origin = info.origin.position
            self.occ_map_height = info.height
            self.occ_map_width = info.width
        except rospy.ServiceException as e:
            rospy.logerr("Failed to get static occupancy map. Ignoring map...")
            self.map_initialized = False

    def is_driveable(self, pose):
        if self.map_initialized:
            x = np.sqrt((pose.position.x - self.occ_map_origin.x) ** 2)
            y = np.sqrt((pose.position.y - self.occ_map_origin.y) ** 2)
            if int(y / self.occ_map_res) >= self.occ_map.shape[0] or \
                    self.occ_map_width - int(x / self.occ_map_res) >= self.occ_map.shape[1]:
                return False
            return 0 <= self.occ_map[int(y / self.occ_map_res)][self.occ_map_width - int(x / self.occ_map_res)] < 1
        else:
            return True

    def is_collision_free(self, pose):
        with self.god_map.get_data(identifier.giskard + ['state_validator_lock']):
            # Get current joint states
            old_js = deepcopy(get_robot_joint_state(self.collision_scene))
            # Calc IK for navigating to given state and ...
            results = []
            for i in range(0, self.ik_sampling):
                if self.debug:
                    s_s = current_milli_time()
                # override on current joint states.
                self.set_tip_link(deepcopy(old_js), pose)
                if self.debug:
                    s_c = current_milli_time()
                results.append(
                    self.collision_scene.are_robot_links_external_collision_free(self.collision_link_names,
                                                                                 dist=self.dist))
                if self.debug:
                    e_c = current_milli_time()
                # Reset joint state
                self.publish_robot_state()
                update_robot_state(self.collision_scene, old_js)
                if self.debug:
                    e_s = current_milli_time()
                    self.debug_times_cc.append(e_c - s_c)
                    self.debug_times.append(e_s - s_s)
                    rospy.loginfo(f'State Validator {self.__class__.__name__}: '
                                  f'CC time: {np.mean(np.array(self.debug_times_cc))} ms. '
                                  f'State Update time: {np.mean(np.array(self.debug_times)) - np.mean(np.array(self.debug_times_cc))} ms.')
            return any(results)

    def publish_robot_state(self):
        if self.publisher is not None:
            self.publisher.update()

    def get_furthest_normal(self, pose):
        # Get current joint states
        old_js = deepcopy(get_robot_joint_state(self.collision_scene))
        # Calc IK for navigating to given state and ...
        # override on current joint states.
        self.set_tip_link(old_js, pose)
        # Check if kitchen is colliding with robot
        result = self.collision_scene.get_furthest_normal(self.collision_link_names)
        # Reset joint state
        update_robot_state(self.collision_scene, old_js)
        return result

    def get_closest_collision_distance(self, pose, link_names):
        # Get current joint states
        old_js = deepcopy(get_robot_joint_state(self.collision_scene))
        # Calc IK for navigating to given state and ...
        # override on current joint states.
        self.set_tip_link(old_js, pose)
        # Check if kitchen is colliding with robot
        collision = self.collision_scene.get_furthest_collision(link_names)[0]
        # Reset joint state
        update_robot_state(self.collision_scene, old_js)
        return collision.contact_distance

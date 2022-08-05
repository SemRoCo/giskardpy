import threading
import time
from copy import deepcopy

import numpy as np
import rospy
from nav_msgs.srv import GetMap

from giskardpy import identifier
from giskardpy.data_types import PrefixName
from giskardpy.path_planning.ik import PyBulletIK
from giskardpy.tree.behaviors.visualization import VisualizationBehavior


def current_milli_time():
    return round(time.time() * 1000)


def update_joint_state(js, new_js):
    if any(map(lambda e: type(e) != PrefixName, new_js)):
        raise Exception('oi, there are no PrefixNames in yer new_js >:(!')
    js.update((k, new_js[k]) for k in js.keys() and new_js.keys())


def update_robot_state(collision_scene, state):
    update_joint_state(collision_scene.world.state, state)
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


class GiskardRobotBulletCollisionChecker(AbstractStateValidator):

    def __init__(self, is_3D, root_link, tip_link, collision_scene, god_map, ik=None, ik_sampling=1, ignore_orientation=False,
                 publish=False, dist=0.0):
        super().__init__(is_3D)
        self.giskard_lock = threading.Lock()
        if ik is None:
            self.ik = PyBulletIK(root_link, tip_link)
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
        self.ignore_orientation = ignore_orientation
        # self.collision_objects = GiskardPyBulletAABBCollision(self.robot, collision_scene, tip_link)
        self.collision_link_names = collision_scene.world.get_children_with_collisions_from_link(self.tip_link)
        self.publisher = None
        self.init_map()
        if publish:
            self.publisher = VisualizationBehavior('motion planning object publisher', ensure_publish=False)
            self.publisher.setup(10)

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
        with self.god_map.get_data(identifier.rosparam + ['state_validator_lock']):
            # Get current joint states
            old_js = deepcopy(self.collision_scene.robot.state)
            # Calc IK for navigating to given state and ...
            results = []
            for i in range(0, self.ik_sampling):
                if self.debug:
                    s_s = current_milli_time()
                state_ik = self.ik.get_ik(deepcopy(self.collision_scene.robot.state), pose)
                # override on current joint states.
                update_robot_state(self.collision_scene, state_ik)
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
        old_js = deepcopy(self.collision_scene.robot.state)
        # Calc IK for navigating to given state and ...
        state_ik = self.ik.get_ik(old_js, pose)
        # override on current joint states.
        update_robot_state(self.collision_scene, state_ik)
        # Check if kitchen is colliding with robot
        result = self.collision_scene.get_furthest_normal(self.collision_link_names)
        # Reset joint state
        update_robot_state(self.collision_scene, old_js)
        return result

    def get_closest_collision_distance(self, pose, link_names):
        # Get current joint states
        old_js = deepcopy(self.collision_scene.robot.state)
        # Calc IK for navigating to given state and ...
        state_ik = self.ik.get_ik(old_js, pose)
        # override on current joint states.
        update_robot_state(self.collision_scene, state_ik)
        # Check if kitchen is colliding with robot
        collision = self.collision_scene.get_furthest_collision(link_names)[0]
        # Reset joint state
        update_robot_state(self.collision_scene, old_js)
        return collision.contact_distance

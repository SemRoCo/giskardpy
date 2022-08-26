import threading
import time
from copy import deepcopy

import numpy as np
import rospy

from giskardpy import identifier
from giskardpy.data_types import PrefixName
from giskardpy.model.pybullet_syncer import PyBulletMotionValidationIDs, PyBulletRayTester, PyBulletBoxSpace
from giskardpy.path_planning.state_validator import update_robot_state, get_robot_joint_state
from giskardpy.utils.tfwrapper import pose_stamped_to_list, pose_to_list, interpolate_pose


# def get_simple_environment_objects_as_urdf(god_map, env_name='kitchen', robot_name):
#     world = god_map.get_data(identifier.world)
#     environment_objects = list()
#     for g_n in world.group_names:
#         if env_name == g_n or robot_name == g_n:
#             continue
#         else:
#             if len(world.groups[g_n].links) == 1:
#                 environment_objects.append(world.groups[g_n].links[g_n].as_urdf())
#     return environment_objects


def get_simple_environment_object_names(god_map, robot_name, env_name='kitchen'):
    world = god_map.get_data(identifier.world)
    environment_objects = list()
    for g_n in world.group_names:
        if env_name == g_n or robot_name == g_n:
            continue
        else:
            if len(world.groups[g_n].links) == 1:
                environment_objects.append(g_n)
    return environment_objects


def current_milli_time():
    return round(time.time() * 1000)


class AbstractMotionValidator():
    """
    This class ensures that every Planner in OMPL makes the same assumption for
    planning edges in the resulting path. The state validator can be if needed
    deactivated by passing ignore_state_validator as True. The real motion checking must be implemented
    by overwriting the function ompl_check_motion.
    """

    def __init__(self, tip_link, god_map, ignore_state_validator=False):
        self.god_map = god_map
        self.state_validator = None
        self.tip_link = tip_link
        self.ignore_state_validator = ignore_state_validator

    def checkMotion(self, s1, s2):
        with self.god_map.get_data(identifier.giskard + ['motion_validator_lock']):
            res_a = self.check_motion(s1, s2)
            res_b = self.check_motion(s2, s1)
            if self.ignore_state_validator:
                return res_a and res_b
            else:
                return res_a and res_b and \
                       self.state_validator.is_collision_free(s1) and \
                       self.state_validator.is_collision_free(s2)

    def checkMotionTimed(self, s1, s2):
        with self.god_map.get_data(identifier.giskard + ['motion_validator_lock']):
            c1, f1 = self.check_motion_timed(s1, s2)
            if not self.ignore_state_validator:
                c1 = c1 and self.state_validator.is_collision_free(s1)
            c2, f2 = self.check_motion_timed(s2, s1)
            if not self.ignore_state_validator:
                c2 = c2 and self.state_validator.is_collision_free(s2)
            if not c1:
                time = max(0, f1 - 0.01)
                last_valid = interpolate_pose(s1, s2, time)
                return False, last_valid, time
            elif not c2:
                # calc_f = 1.0 - f2
                # colliding = True
                # while colliding:
                #    calc_f -= 0.01
                #    colliding, new_f = self.just_check_motion(s1, self.get_last_valid(s1, s2, calc_f))
                # last_valid = self.get_last_valid(s1, s2, max(0, calc_f-0.05))
                return False, s1, 0
            else:
                return True, s2, 1

    def check_motion(self, s1, s2):
        raise Exception('Please overwrite me')

    def check_motion_timed(self, s1, s2):
        raise Exception('Please overwrite me')

    def clear(self):
        pass


class SimpleRayMotionValidator(AbstractMotionValidator):

    def __init__(self, collision_scene, tip_link, god_map, debug=False, js=None, ignore_state_validator=None):
        AbstractMotionValidator.__init__(self, tip_link, god_map, ignore_state_validator=ignore_state_validator)
        self.hitting = {}
        self.debug = debug
        self.js = deepcopy(js)
        self.debug_times = list()
        self.raytester_lock = threading.Lock()
        environment_object_names = get_simple_environment_object_names(self.god_map, collision_scene.robot.name)
        self.collision_scene = collision_scene
        self.collision_link_names = self.collision_scene.world.get_children_with_collisions_from_link(self.tip_link)
        pybulletenv = PyBulletMotionValidationIDs(self.god_map.get_data(identifier.collision_scene),
                                                  environment_object_names=environment_object_names,
                                                  moving_links=self.collision_link_names)
        self.raytester = PyBulletRayTester(pybulletenv=pybulletenv)

    def check_motion(self, s1, s2):
        with self.raytester_lock:
            res, _, _, _ = self._ray_test_wrapper(s1, s2)
            return res

    def check_motion_timed(self, s1, s2):
        with self.raytester_lock:
            res, _, _, f = self._ray_test_wrapper(s1, s2)
            return res, f

    def _ray_test_wrapper(self, s1, s2):
        if self.debug:
            s = current_milli_time()
        self.raytester.pre_ray_test()
        collision_free, coll_links, dists, fractions = self._check_motion(s1, s2)
        self.raytester.post_ray_test()
        if self.debug:
            e = current_milli_time()
            self.debug_times.append(e - s)
            rospy.loginfo(f'Motion Validator {self.__class__.__name__}: '
                          f'Raytester: {self.raytester.__class__.__name__}: '
                          f'Summed time: {np.sum(np.array(self.debug_times))} ms.')
        return collision_free, coll_links, dists, fractions

    @profile
    def _check_motion(self, s1, s2):
        # Shoot ray from start to end pose and check if it intersects with the kitchen,
        # if so return false, else true.
        query_b = [[s1.position.x, s1.position.y, s1.position.z]]
        query_e = [[s2.position.x, s2.position.y, s2.position.z]]
        collision_free, coll_links, dists, fractions = self.raytester.ray_test_batch(query_b, query_e)
        return collision_free, coll_links, dists, min(fractions)


class ObjectRayMotionValidator(SimpleRayMotionValidator):

    def __init__(self, collision_scene, tip_link, object_in_motion, state_validator, god_map, debug=False,
                 js=None, ignore_state_validator=False):
        SimpleRayMotionValidator.__init__(self, collision_scene, tip_link, god_map, debug=debug,
                                          js=js, ignore_state_validator=ignore_state_validator)
        self.state_validator = state_validator
        self.object_in_motion = object_in_motion

    @profile
    def _check_motion(self, s1, s2):
        # Shoot ray from start to end pose and check if it intersects with the kitchen,
        # if so return false, else true.
        old_js = deepcopy(get_robot_joint_state(self.collision_scene))
        # s = 0.
        # for j_n, v in state1.items():
        #    v2 = self.state_validator.ik.get_ik(old_js, s1)[j_n].position
        #    n = abs(v.position - v2)
        #    if n != 0:
        #        rospy.logerr(f'joint_name: {j_n}: first: {v.position}, second: {v2}, diff: {n}')
        #    s += n
        self.state_validator.set_tip_link(old_js, s1)
        query_b = self.collision_scene.get_aabb_collisions(self.collision_link_names).get_points()
        # s = 0.
        # for j_n, v in state2.items():
        #    v2 = self.state_validator.ik.get_ik(old_js, s2)[j_n].position
        #    n = abs(v.position - v2)
        #    if n != 0:
        #        rospy.logerr(f'joint_name: {j_n}: first: {v.position}, second: {v2}, diff: {n}')
        #    s += n
        self.state_validator.set_tip_link(old_js, s2)
        query_e = self.collision_scene.get_aabb_collisions(self.collision_link_names).get_points()
        update_robot_state(self.collision_scene, old_js)
        collision_free, coll_links, dists, fractions = self.raytester.ray_test_batch(query_b, query_e)
        return collision_free, coll_links, dists, min(fractions)


class CompoundBoxMotionValidator(AbstractMotionValidator):

    def __init__(self, collision_scene, tip_link, object_in_motion, state_validator, god_map, js=None, links=None):
        super(CompoundBoxMotionValidator, self).__init__(tip_link, god_map)
        self.collision_scene = collision_scene
        self.state_validator = state_validator
        self.object_in_motion = object_in_motion
        environment_object_names = get_simple_environment_object_names(self.god_map, self.god_map.get_data(
            identifier.robot_group_name))
        self.collision_link_names = self.collision_scene.world.get_children_with_collisions_from_link(self.tip_link)
        pybulletenv = PyBulletMotionValidationIDs(self.god_map.get_data(identifier.collision_scene),
                                                  environment_object_names=environment_object_names,
                                                  moving_links=self.collision_link_names)
        self.box_space = PyBulletBoxSpace(self.collision_scene.world, self.object_in_motion, 'map', pybulletenv)
        # self.collision_points = GiskardPyBulletAABBCollision(object_in_motion, collision_scene, tip_link, links=links)

    @profile
    def check_motion_old(self, s1, s2):
        old_js = deepcopy(get_robot_joint_state(self.collision_scene))
        ret = True
        for collision_link_name in self.collision_link_names:
            self.state_validator.set_tip_link(old_js, s1)
            query_b = pose_stamped_to_list(self.collision_scene.get_pose(collision_link_name))
            self.state_validator.set_tip_link(old_js, s2)
            query_e = pose_stamped_to_list(self.collision_scene.get_pose(collision_link_name))
            start_positions = [query_b[0]]
            end_positions = [query_e[0]]
            collision_object = self.collision_scene.get_aabb_info(collision_link_name)
            min_size = np.max(np.abs(np.array(collision_object.d) - np.array(collision_object.u)))
            if self.box_space.is_colliding([min_size], start_positions, end_positions):
                # self.object_in_motion.state = self.collision_scene.world.state
                ret = False
                break
        update_robot_state(self.collision_scene, old_js)
        return ret

    @profile
    def get_box_params(self, s1, s2):
        old_js = deepcopy(get_robot_joint_state(self.collision_scene))
        self.state_validator.set_tip_link(old_js, s1)
        start_positions = list()
        end_positions = list()
        min_sizes = list()
        for collision_link_name in self.collision_link_names:
            start_positions.append(pose_stamped_to_list(self.collision_scene.get_pose(collision_link_name))[0])
        self.state_validator.set_tip_link(old_js, s2)
        for collision_link_name in self.collision_link_names:
            end_positions.append(pose_stamped_to_list(self.collision_scene.get_pose(collision_link_name))[0])
            collision_object = self.collision_scene.get_aabb_info(collision_link_name)
            min_size = np.max(np.abs(np.array(collision_object.d) - np.array(collision_object.u)))
            min_sizes.append(min_size)
        update_robot_state(self.collision_scene, old_js)
        return min_sizes, start_positions, end_positions

    @profile
    def check_motion(self, s1, s2):
        return not self.box_space.is_colliding(*self.get_box_params(s1, s2))

    @profile
    def check_motion_timed(self, s1, s2):
        m, s, e = self.get_box_params(s1, s2)
        c, f = self.box_space.is_colliding_timed(m, s, e, pose_to_list(s1)[0], pose_to_list(s2)[0])
        return not c, f

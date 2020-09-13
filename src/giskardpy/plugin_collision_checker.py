import itertools
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Lock

import rospy
from py_trees import Status
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest

import giskardpy.identifier as identifier
from giskardpy import pybullet_wrapper
from giskardpy.plugin import GiskardBehavior


class CollisionChecker(GiskardBehavior):
    def __init__(self, name):
        super(CollisionChecker, self).__init__(name)
        # self.default_min_dist = self.get_god_map().safe_get_data(identifier.default_collision_avoidance_distance)
        self.map_frame = self.get_god_map().get_data(identifier.map_frame)
        self.lock = Lock()
        self.object_js_subs = {}  # JointState subscribers for articulated world objects
        self.object_joint_states = {}  # JointStates messages for articulated world objects
        self.get_god_map().set_data(identifier.added_collision_checks, {})

    def setup(self, timeout=10.0):
        super(CollisionChecker, self).setup(timeout)
        # self.pub_collision_marker = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)
        self.srv_activate_rendering = rospy.Service(u'~render', SetBool, self.activate_rendering)
        rospy.sleep(.5)
        return True

    def activate_rendering(self, data):
        """
        :type data: SetBoolRequest
        :return:
        """
        pybullet_wrapper.render = data.data
        if data.data:
            pybullet_wrapper.activate_rendering()
        else:
            pybullet_wrapper.deactivate_rendering()
        return SetBoolResponse()

    def initialise(self):
        collision_goals = self.get_god_map().get_data(identifier.collision_goal)
        # max_distance = self.get_god_map().get_data(identifier.maximum_collision_threshold)
        external_distances = self.get_god_map().get_data(identifier.external_collision_avoidance_distance)
        self_distances = self.get_god_map().get_data(identifier.self_collision_avoidance_distance)
        default_distance = max(external_distances.default_factory()[u'soft_threshold'],
                               self_distances.default_factory()[u'soft_threshold'])
        max_distances = defaultdict(lambda: default_distance)

        for link_name in self.get_robot().get_links_with_collision():
            controlled_parent_joint = self.get_robot().get_controlled_parent_joint(link_name)
            distance = external_distances[controlled_parent_joint][u'soft_threshold']
            for child_link_name in self.get_robot().get_directly_controllable_collision_links(controlled_parent_joint):
                max_distances[child_link_name] = distance

        for link_name in self_distances:
            distance = self_distances[link_name][u'soft_threshold']
            if link_name in max_distances:
                max_distances[link_name] = max(distance, max_distances[link_name])
            else:
                max_distances[link_name] = distance

        added_checks = self.get_god_map().get_data(identifier.added_collision_checks)
        for link_name, distance in added_checks.items():
            if link_name in max_distances:
                max_distances[link_name] = max(distance, max_distances[link_name])
            else:
                max_distances[link_name] = distance



        self.collision_matrix = self.get_world().collision_goals_to_collision_matrix(deepcopy(collision_goals), max_distances)

        super(CollisionChecker, self).initialise()

    def update(self):
        """
        Computes closest point info for all robot links and safes it to the god map.
        """
        collisions = self.get_world().check_collisions(self.collision_matrix)
        self.god_map.safe_set_data(identifier.closest_point, collisions)
        return Status.RUNNING

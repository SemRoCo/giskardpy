import itertools
from collections import defaultdict
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
        external_distance_thresholds = self.get_god_map().get_data(identifier.external_collision_avoidance_distance)
        self_distance_thresholds = self.get_god_map().get_data(identifier.self_collision_avoidance_distance)

        # trigger default factory... FIXME pls
        external_distance_thresholds[0]
        self_distance_thresholds[0]

        max_distance = 0
        for thresholds in itertools.chain(external_distance_thresholds.values(), self_distance_thresholds.values()):
            max_distance = max(max_distance, thresholds[u'soft_threshold']) # FIXME get rid of hardcoded string

        max_distances = defaultdict(lambda: max_distance)

        self.collision_matrix = self.get_world().collision_goals_to_collision_matrix(collision_goals, max_distances)

        super(CollisionChecker, self).initialise()

    def update(self):
        """
        Computes closest point info for all robot links and safes it to the god map.
        """
        collisions = self.get_world().check_collisions(self.collision_matrix)
        self.god_map.set_data(identifier.closest_point, collisions)
        return Status.RUNNING

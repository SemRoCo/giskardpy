from multiprocessing import Lock

import rospy
from py_trees import Status
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest

import giskardpy.identifier as identifier
from giskardpy.model import pybullet_wrapper
from giskardpy.tree.plugin import GiskardBehavior


class CollisionChecker(GiskardBehavior):
    def __init__(self, name):
        super(CollisionChecker, self).__init__(name)
        self.map_frame = 'map'
        self.lock = Lock()
        self.object_js_subs = {}  # JointState subscribers for articulated world objects
        self.object_joint_states = {}  # JointStates messages for articulated world objects
        self.get_god_map().set_data(identifier.added_collision_checks, {})

    def setup(self, timeout=10.0):
        super(CollisionChecker, self).setup(timeout)
        # self.collision_scene.init_collision_matrix(RobotName)
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
        self.collision_scene.update_collision_environment()

        #self.get_god_map().set_data(identifier.collision_matrix, self.collision_matrix)
        #self.get_god_map().set_data(identifier.collision_list_size, self.collision_list_size)

        super(CollisionChecker, self).initialise()

    @profile
    def update(self):
        """
        Computes closest point info for all robot links and safes it to the god map.
        """
        self.collision_scene.update_collision_checker()
        return Status.RUNNING

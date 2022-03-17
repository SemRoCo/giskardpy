from time import time

from py_trees import Behaviour, Blackboard

from giskardpy import identifier
from giskardpy.god_map import GodMap
from giskardpy.model.world import WorldTree
from giskardpy.utils.time_collector import TimeCollector


class GiskardBehavior(Behaviour):
    time_collector: TimeCollector

    def __init__(self, name):
        self.god_map = Blackboard().god_map  # type: GodMap
        self.time_collector = self.god_map.unsafe_get_data(identifier.timer_collector)
        self.world = self.get_god_map().unsafe_get_data(identifier.world)  # type: WorldTree
        super(GiskardBehavior, self).__init__(name)

    @property
    def traj_time_in_sec(self):
        return self.god_map.unsafe_get_data(identifier.time) * self.god_map.unsafe_get_data(identifier.sample_period)

    def get_god_map(self):
        """
        :rtype: giskardpy.god_map.GodMap
        """
        return self.god_map

    def get_runtime(self):
        return time() - self.get_blackboard().runtime

    @property
    def tree(self):
        """
        :rtype: giskardpy.tree.garden.TreeManager
        """
        return self.god_map.unsafe_get_data(identifier.tree_manager)

    @property
    def collision_scene(self):
        """
        :rtype: giskardpy.model.collision_world_syncer.CollisionWorldSynchronizer
        """
        return self.god_map.unsafe_get_data(identifier.collision_scene)

    @collision_scene.setter
    def collision_scene(self, value):
        self.god_map.unsafe_set_data(identifier.collision_scene, value)

    def robot(self, robot_name=''):
        """
        :rtype: giskardpy.model.world.SubWorldTree
        """
        return self.world.groups[robot_name]

    def robot_names(self):
        return self.world.god_map.get_data(identifier.rosparam + ['namespaces'])

    def get_world(self):
        """
        :rtype: giskardpy.model.world.WorldTree
        """
        return self.world

    def unsafe_get_world(self):
        """
        :rtype: giskardpy.model.world.WorldTree
        """
        return self.world

    def raise_to_blackboard(self, exception):
        Blackboard().set('exception', exception)

    def get_blackboard(self):
        return Blackboard()

    def get_blackboard_exception(self):
        return self.get_blackboard().get('exception')

    def clear_blackboard_exception(self):
        self.get_blackboard().set('exception', None)

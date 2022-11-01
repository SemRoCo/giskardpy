from copy import deepcopy

from py_trees import Status

from giskardpy import identifier
from giskardpy.model.collision_world_syncer import Collisions
from giskardpy.model.trajectory import Trajectory
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class CleanUp(GiskardBehavior):
    @profile
    def __init__(self, name):
        super().__init__(name)

    @profile
    def initialise(self):
        self.god_map.clear_cache()
        self.god_map.get_data(identifier.giskard)._reset_config()
        self.god_map.set_data(identifier.goal_msg, None)
        self.world.fast_all_fks = None
        self.collision_scene.reset_cache()
        self.god_map.set_data(identifier.closest_point, Collisions(self.god_map, 1))
        # self.get_god_map().safe_set_data(identifier.closest_point, None)
        self.god_map.set_data(identifier.time, 1)

        # to reverse update godmap changes
        # self.get_god_map().set_data(identifier.giskard, deepcopy(self.rosparams))
        self.world.sync_with_paramserver()
        self.god_map.set_data(identifier.next_move_goal, None)
        if hasattr(self.get_blackboard(), 'runtime'):
            del self.get_blackboard().runtime

    def update(self):
        return Status.SUCCESS


class CleanUpPlanning(CleanUp):
    def initialise(self):
        super().initialise()
        self.god_map.set_data(identifier.fill_trajectory_velocity_values, None)


class CleanUpBaseController(CleanUp):
    pass

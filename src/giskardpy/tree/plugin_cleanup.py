from copy import deepcopy

from py_trees import Status

from giskardpy import identifier
from giskardpy.data_types import Trajectory, Collisions
from giskardpy.tree.plugin import GiskardBehavior


class CleanUp(GiskardBehavior):
    def __init__(self, name):
        super(CleanUp, self).__init__(name)
        # FIXME this is the smallest hack to reverse (some) update godmap changes, constraints need some kind of finalize
        self.general_options = deepcopy(self.get_god_map().get_data(identifier.general_options))

    def initialise(self):
        self.get_god_map().clear_cache()
        self.god_map.set_data(identifier.goal_msg, None)
        self.world.fast_all_fks = None
        self.collision_scene.reset_cache()
        self.get_god_map().set_data(identifier.closest_point, Collisions(self.world, 1))
        # self.get_god_map().safe_set_data(identifier.closest_point, None)
        self.get_god_map().set_data(identifier.time, 1)
        current_js = deepcopy(self.get_god_map().get_data(identifier.joint_states))
        trajectory = Trajectory()
        trajectory.set(0, current_js)
        self.get_god_map().set_data(identifier.trajectory, trajectory)
        trajectory = Trajectory()
        self.get_god_map().set_data(identifier.debug_trajectory, trajectory)
        # to reverse update godmap changes
        self.get_god_map().set_data(identifier.general_options, deepcopy(self.general_options))
        self.get_god_map().set_data(identifier.next_move_goal, None)
        if hasattr(self.get_blackboard(), 'runtime'):
            del self.get_blackboard().runtime

    def update(self):
        return Status.SUCCESS

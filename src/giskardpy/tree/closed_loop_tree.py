import pydot
from py_trees import Sequence
from py_trees_ros.trees import BehaviourTree

import giskardpy
from giskard_msgs.msg import MoveAction
from giskardpy import identifier, RobotName
from giskardpy.god_map import GodMap
from giskardpy.tree.async_composite import PluginBehavior
from giskardpy.tree.cleanup import CleanUp
from giskardpy.tree.collision_checker import CollisionChecker
from giskardpy.tree.goal_reached import GoalReachedPlugin
from giskardpy.tree.instantaneous_controller import ControllerPlugin
from giskardpy.tree.kinematic_sim import KinSimPlugin
from giskardpy.tree.log_debug_expressions import LogDebugExpressionsPlugin
from giskardpy.tree.log_trajectory import LogTrajPlugin
from giskardpy.tree.max_trajectory_length import MaxTrajectoryLength
from giskardpy.tree.open_loop_tree import OpenLoopTree
from giskardpy.tree.send_result import SendResult
from giskardpy.tree.sync_configuration2 import SyncConfiguration2
from giskardpy.tree.time import TimePlugin
from giskardpy.tree.tree_manager import TreeManager
from giskardpy.utils.utils import get_all_classes_in_package


class ClosedLoopTree(OpenLoopTree):

    def grow_giskard(self):
        root = Sequence('Giskard')
        root.add_child(self.grow_wait_for_goal())
        root.add_child(CleanUp('cleanup'))
        root.add_child(self.grow_process_goal())
        root.add_child(SendResult('send result', self.action_server_name, MoveAction))
        return root

    # def grow_sync_branch(self):
    #     sync = Sequence('Synchronize')
    #     sync.add_child(WorldUpdater('update world'))
    #     sync.add_child(running_is_success(SyncConfiguration)('update robot configuration', RobotName))
    #     sync.add_child(SyncLocalization('update robot localization', RobotName))
    #     sync.add_child(TFPublisher('publish tf', **self.god_map.get_data(identifier.TFPublisher)))
    #     sync.add_child(CollisionSceneUpdater('update collision scene'))
    #     sync.add_child(running_is_success(VisualizationBehavior)('visualize collision scene'))
    #     return sync

    def grow_planning3(self):
        planning_3 = Sequence('planning III', sleep=0)
        planning_3.add_child(self.grow_planning4())
        return planning_3

    def grow_planning4(self):
        planning_4 = PluginBehavior('planning IIII', hz=True, sleep=0)
        action_servers = self.god_map.get_data(identifier.robot_interface)
        behaviors = get_all_classes_in_package(giskardpy.tree)
        for i, (execution_action_server_name, params) in enumerate(action_servers.items()):
            C = behaviors[params['plugin']]
            del params['plugin']
            planning_4.add_plugin(C(execution_action_server_name, **params))
        planning_4.add_plugin(SyncConfiguration2('update robot configuration', RobotName))
        planning_4.add_plugin(LogTrajPlugin('log'))
        if self.god_map.get_data(identifier.collision_checker) is not None:
            planning_4.add_plugin(CollisionChecker('collision checker'))
        planning_4.add_plugin(ControllerPlugin('controller'))
        planning_4.add_plugin(KinSimPlugin('kin sim'))

        if self.god_map.get_data(identifier.PlotDebugTrajectory_enabled):
            planning_4.add_plugin(LogDebugExpressionsPlugin('log lba'))
        # planning_4.add_plugin(WiggleCancel('wiggle'))
        # planning_4.add_plugin(LoopDetector('loop detector'))
        planning_4.add_plugin(GoalReachedPlugin('goal reached'))
        planning_4.add_plugin(TimePlugin('time'))
        if self.god_map.get_data(identifier.MaxTrajectoryLength_enabled):
            kwargs = self.god_map.get_data(identifier.MaxTrajectoryLength)
            planning_4.add_plugin(MaxTrajectoryLength('traj length check', **kwargs))
        return planning_4

from py_trees import Sequence
from py_trees_ros.trees import BehaviourTree

from giskard_msgs.msg import MoveAction
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.send_result import SendResult
from giskardpy.tree.branches.clean_up_control_loop import CleanupControlLoop
from giskardpy.tree.branches.post_processing import PostProcessing
from giskardpy.tree.branches.prepare_control_loop import PrepareControlLoop
from giskardpy.tree.branches.process_goal import ProcessGoal
from giskardpy.tree.branches.wait_for_goal import WaitForGoal
from giskardpy.tree.control_modes import ControlModes
from giskardpy.tree.decorators import failure_is_success


class GiskardBT(BehaviourTree):
    wait_for_goal: WaitForGoal
    prepare_control_loop: PrepareControlLoop
    process_goal: ProcessGoal
    post_processing: PostProcessing
    cleanup_control_loop: CleanupControlLoop

    def __init__(self, control_mode: ControlModes):
        god_map.tree_manager.control_mode = control_mode
        # TODO reject invalid control mode
        # raise KeyError(f'Robot interface mode \'{self._control_mode}\' is not supported.')
        root = Sequence('Giskard')
        self.wait_for_goal = WaitForGoal()
        self.prepare_control_loop = failure_is_success(PrepareControlLoop)()
        self.process_goal = failure_is_success(ProcessGoal)()
        self.post_processing = failure_is_success(PostProcessing)()
        self.cleanup_control_loop = CleanupControlLoop()

        root.add_child(self.wait_for_goal)
        root.add_child(self.prepare_control_loop)
        root.add_child(self.process_goal)
        root.add_child(self.cleanup_control_loop)
        root.add_child(self.post_processing)
        root.add_child(SendResult('send result',
                                  god_map.giskard.action_server_name,
                                  MoveAction))
        super().__init__(root)

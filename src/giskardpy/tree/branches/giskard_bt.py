from py_trees import Sequence
from py_trees_ros.trees import BehaviourTree

from giskard_msgs.msg import MoveAction
from giskardpy.exceptions import GiskardException
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.send_result import SendResult
from giskardpy.tree.branches.clean_up_control_loop import CleanupControlLoop
from giskardpy.tree.branches.control_loop import ControlLoop
from giskardpy.tree.branches.post_processing import PostProcessing
from giskardpy.tree.branches.prepare_control_loop import PrepareControlLoop
from giskardpy.tree.branches.send_trajectories import ExecuteTraj
from giskardpy.tree.branches.wait_for_goal import WaitForGoal
from giskardpy.tree.control_modes import ControlModes
from giskardpy.tree.decorators import failure_is_success


class GiskardBT(BehaviourTree):
    wait_for_goal: WaitForGoal
    prepare_control_loop: PrepareControlLoop
    post_processing: PostProcessing
    cleanup_control_loop: CleanupControlLoop
    control_loop_branch: ControlLoop

    def __init__(self, control_mode: ControlModes):
        god_map.tree_manager.control_mode = control_mode
        if control_mode not in ControlModes:
            raise AttributeError(f'Control mode {control_mode} doesn\'t exist.')
        root = Sequence('Giskard')
        self.wait_for_goal = WaitForGoal()
        self.prepare_control_loop = failure_is_success(PrepareControlLoop)()
        self.control_loop_branch = failure_is_success(ControlLoop)(projection=not god_map.is_closed_loop())
        self.post_processing = failure_is_success(PostProcessing)()
        self.cleanup_control_loop = CleanupControlLoop()
        if god_map.is_planning():
            self.execute_traj = failure_is_success(ExecuteTraj)()

        root.add_child(self.wait_for_goal)
        root.add_child(self.prepare_control_loop)
        root.add_child(self.control_loop_branch)
        root.add_child(self.cleanup_control_loop)
        if god_map.is_planning():
            root.add_child(self.execute_traj)
        root.add_child(self.post_processing)
        root.add_child(SendResult('send result',
                                  god_map.giskard.action_server_name,
                                  MoveAction))
        super().__init__(root)

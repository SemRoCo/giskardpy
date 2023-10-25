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
from giskardpy.utils.decorators import toggle_on, toggle_off


class GiskardBT(BehaviourTree):
    wait_for_goal: WaitForGoal
    prepare_control_loop: PrepareControlLoop
    post_processing: PostProcessing
    cleanup_control_loop: CleanupControlLoop
    control_loop_branch: ControlLoop
    root: Sequence
    execute_traj: ExecuteTraj

    def __init__(self, control_mode: ControlModes):
        god_map.tree_manager.control_mode = control_mode
        if control_mode not in ControlModes:
            raise AttributeError(f'Control mode {control_mode} doesn\'t exist.')
        self.root = Sequence('Giskard')
        self.wait_for_goal = WaitForGoal()
        self.prepare_control_loop = failure_is_success(PrepareControlLoop)()
        self.control_loop_branch = failure_is_success(ControlLoop)()
        if god_map.is_closed_loop():
            self.control_loop_branch.add_closed_loop_behaviors()
        else:
            self.control_loop_branch.add_projection_behaviors()

        self.post_processing = failure_is_success(PostProcessing)()
        self.cleanup_control_loop = CleanupControlLoop()
        if god_map.is_planning():
            self.execute_traj = failure_is_success(ExecuteTraj)()

        self.root.add_child(self.wait_for_goal)
        self.root.add_child(self.prepare_control_loop)
        self.root.add_child(self.control_loop_branch)
        self.root.add_child(self.cleanup_control_loop)
        self.root.add_child(self.post_processing)
        self.root.add_child(SendResult('send result',
                                       god_map.giskard.action_server_name,
                                       MoveAction))
        super().__init__(self.root)
        self.switch_to_execution()

    @toggle_on('visualization_mode')
    def turn_on_visualization(self):
        self.wait_for_goal.publish_state.add_visualization_marker_behavior()
        self.control_loop_branch.publish_state.add_visualization_marker_behavior()

    @toggle_off('visualization_mode')
    def turn_off_visualization(self):
        self.wait_for_goal.publish_state.remove_visualization_marker_behavior()
        self.control_loop_branch.publish_state.remove_visualization_marker_behavior()

    @toggle_on('projection_mode')
    def switch_to_projection(self):
        if god_map.is_planning():
            self.root.remove_child(self.execute_traj)
        elif god_map.is_closed_loop():
            self.control_loop_branch.switch_to_projection()
        self.cleanup_control_loop.add_reset_world_state()

    @toggle_off('projection_mode')
    def switch_to_execution(self):
        if god_map.is_planning():
            self.root.insert_child(self.execute_traj, -2)
        elif god_map.is_closed_loop():
            self.control_loop_branch.switch_to_closed_loop()
        self.cleanup_control_loop.remove_reset_world_state()

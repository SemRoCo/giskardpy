from py_trees import Sequence, Selector

from giskard_msgs.msg import MoveFeedback
from giskardpy import identifier
from giskardpy.god_map_user import GodMapWorshipper
from giskardpy.tree.behaviors.exception_to_execute import ClearBlackboardException
from giskardpy.tree.behaviors.goal_canceled import GoalCanceled
from giskardpy.tree.behaviors.publish_feedback import PublishFeedback
from giskardpy.tree.behaviors.set_move_result import SetMoveResult
from giskardpy.tree.branches.clean_up_control_loop import CleanupControlLoop
from giskardpy.tree.branches.control_loop import ControlLoopBranch
from giskardpy.tree.decorators import success_is_failure


class ProcessGoal(Selector, GodMapWorshipper):
    control_loop_branch: ControlLoopBranch
    cleanup_control_loop: CleanupControlLoop

    def __init__(self, name: str = 'process goal'):
        super().__init__(name)
        self.cleanup_control_loop = CleanupControlLoop()
        self.control_loop_branch = success_is_failure(ControlLoopBranch)()

        self.add_child(GoalCanceled('goal canceled2', self.god_map.get_data(identifier.action_server_name)))
        self.add_child(success_is_failure(PublishFeedback)('publish feedback1', MoveFeedback.PLANNING))
        # planning_2.add_child(success_is_failure(StartTimer)('start runtime timer'))
        self.add_child(self.control_loop_branch)
        self.add_child(self.cleanup_control_loop)

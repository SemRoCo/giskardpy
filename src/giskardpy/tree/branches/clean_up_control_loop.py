from py_trees import Sequence

from giskard_msgs.msg import MoveFeedback
from giskardpy.god_map_user import GodMapWorshipper
from giskardpy.tree.behaviors.append_zero_velocity import SetZeroVelocity
from giskardpy.tree.behaviors.cleanup import CleanUpPlanning
from giskardpy.tree.behaviors.exception_to_execute import ClearBlackboardException
from giskardpy.tree.behaviors.goal_cleanup import GoalCleanUp
from giskardpy.tree.behaviors.init_qp_controller import InitQPController
from giskardpy.tree.behaviors.log_trajectory import LogTrajPlugin
from giskardpy.tree.behaviors.new_trajectory import NewTrajectory
from giskardpy.tree.behaviors.publish_feedback import PublishFeedback
from giskardpy.tree.behaviors.ros_msg_to_goal import RosMsgToGoal
from giskardpy.tree.behaviors.set_move_result import SetMoveResult
from giskardpy.tree.behaviors.time import TimePlugin
from giskardpy.tree.decorators import success_is_failure, running_is_success


class CleanupControlLoop(Sequence, GodMapWorshipper):
    def __init__(self, name: str = 'clean up control loop'):
        super().__init__(name)
        self.add_child(TimePlugin('increase time plan post processing'))
        self.add_child(SetZeroVelocity('set zero vel 1'))
        self.add_child(LogTrajPlugin('log post processing'))
        self.add_child(GoalCleanUp('clean up goals'))

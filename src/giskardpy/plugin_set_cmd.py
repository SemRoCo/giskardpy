from collections import defaultdict

from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.exceptions import InsolvableException
from giskardpy.plugin_action_server import GetGoal
from giskard_msgs.msg import MoveGoal, CollisionEntry, MoveCmd, MoveResult


class SetCmd(GetGoal):
    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)
        self.current_goal_id = 0
        self.goal = None
        self.sample_period_backup = None
        self.rc_sample_period = self.get_god_map().get_data(identifier.rc_sample_period)

    def initialise(self):
        if self.goal is None:
            self.goal = self.pop_goal()  # type: MoveGoal
            self.get_god_map().set_data(identifier.result_message, MoveResult())
            self.traj = []
            if len(self.goal.cmd_seq) == 0:
                self.raise_to_blackboard(InsolvableException(u'goal empty'))
            if self.goal.type in [MoveGoal.PLAN_AND_EXECUTE, MoveGoal.PLAN_ONLY, MoveGoal.CHECK_REACHABILITY]:
                if self.sample_period_backup is not None:
                    self.get_god_map().set_data(identifier.sample_period, self.sample_period_backup)
                    self.sample_period_backup = None
            else:
                self.raise_to_blackboard(
                    InsolvableException(u'invalid move action goal type: {}'.format(self.goal.type)))
            if self.goal.type == MoveGoal.CHECK_REACHABILITY:
                self.sample_period_backup = self.get_god_map().get_data(identifier.sample_period)
                self.get_god_map().set_data(identifier.sample_period, self.rc_sample_period)
                collision_entry = CollisionEntry()
                collision_entry.type = CollisionEntry.ALLOW_COLLISION
                for cmd in self.goal.cmd_seq:
                    cmd.collisions = [collision_entry]
                self.get_god_map().set_data(identifier.check_reachability, True)
                self.get_god_map().set_data(identifier.execute, False)
            if self.goal.type == MoveGoal.PLAN_AND_EXECUTE:
                self.get_god_map().set_data(identifier.check_reachability, False)
                self.get_god_map().set_data(identifier.execute, True)
            if self.goal.type == MoveGoal.PLAN_ONLY:
                self.get_god_map().set_data(identifier.check_reachability, False)
                self.get_god_map().set_data(identifier.execute, False)

    def update(self):
        # TODO goal checks should probably be its own plugin?
        if self.get_blackboard_exception():
            self.goal = None
            self.get_god_map().set_data(identifier.next_move_goal, None)
            return Status.SUCCESS

        try:
            move_cmd = self.goal.cmd_seq.pop(0)  # type: MoveCmd
            self.get_god_map().set_data(identifier.next_move_goal, move_cmd)
        except IndexError:
            self.goal = None
            self.get_god_map().set_data(identifier.next_move_goal, None)
            return Status.SUCCESS

        return Status.RUNNING
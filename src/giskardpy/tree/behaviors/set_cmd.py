from py_trees import Status

import giskardpy.identifier as identifier
from giskard_msgs.msg import MoveGoal, CollisionEntry, MoveCmd, MoveResult
from giskardpy.exceptions import InvalidGoalException
from giskardpy.tree.behaviors.get_goal import GetGoal
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time
from giskardpy.utils.utils import raise_to_blackboard


class SetCmd(GetGoal):
    @profile
    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)
        self.sample_period_backup = None

    @property
    def goal(self):
        """
        :rtype: MoveGoal
        """
        return self.god_map.get_data(identifier.goal_msg)

    @record_time
    @profile
    def initialise(self):
        if self.goal is None:
            self.god_map.set_data(identifier.goal_msg, self.pop_goal())
            self.number_of_move_cmds = len(self.goal.cmd_seq)
            self.god_map.set_data(identifier.number_of_move_cmds, self.number_of_move_cmds)
            logging.loginfo('Goal has {} move commands(s).'.format(len(self.goal.cmd_seq)))
            self.get_god_map().set_data(identifier.cmd_id, -1)
            empty_result = MoveResult()
            empty_result.error_codes = [MoveResult.ERROR for _ in self.goal.cmd_seq]
            empty_result.error_messages = ['' for _ in self.goal.cmd_seq]
            if not empty_result.error_codes:
                empty_result.error_codes = [MoveResult.ERROR]
                empty_result.error_messages = ['']
            self.traj = []
            if len(self.goal.cmd_seq) == 0:
                empty_result.error_codes = [MoveResult.INVALID_GOAL]
                raise_to_blackboard(InvalidGoalException('goal empty'))
            self.get_god_map().set_data(identifier.result_message, empty_result)
            if self.is_plan(self.goal.type):
                if self.sample_period_backup is not None:
                    self.get_god_map().set_data(identifier.sample_period, self.sample_period_backup)
                    self.sample_period_backup = None
            else:
                error_message = 'Invalid move action goal type: {}'.format(self.goal.type)
                logging.logwarn(error_message)
                logging.logwarn('Goal rejected.')
                raise_to_blackboard(InvalidGoalException(error_message))
            if self.is_check_reachability(self.goal.type):
                self.sample_period_backup = self.get_god_map().get_data(identifier.sample_period)
                self.get_god_map().set_data(identifier.sample_period, self.rc_sample_period)
                collision_entry = CollisionEntry()
                collision_entry.type = CollisionEntry.ALLOW_COLLISION
                for cmd in self.goal.cmd_seq:
                    cmd.collisions = [collision_entry]
                self.get_god_map().set_data(identifier.check_reachability, True)
            else:
                self.get_god_map().set_data(identifier.check_reachability, False)

            self.get_god_map().set_data(identifier.execute, self.is_execute(self.goal.type))
            self.get_god_map().set_data(identifier.skip_failures, self.is_skip_failures(self.goal.type))
            self.get_god_map().set_data(identifier.cut_off_shaking, self.is_cut_off_shaking(self.goal.type))

    def is_plan(self, goal_type, plan_code=1):
        return plan_code in self.get_set_bits(goal_type)

    def is_execute(self, goal_type, execute_code=4):
        return execute_code in self.get_set_bits(goal_type)

    def is_skip_failures(self, goal_type, skip_failures_code=8):
        return skip_failures_code in self.get_set_bits(goal_type)

    def is_check_reachability(self, goal_type, check_reachability_code=2):
        return check_reachability_code in self.get_set_bits(goal_type)

    def is_cut_off_shaking(self, goal_type, cut_off_shaking=16):
        return cut_off_shaking in self.get_set_bits(goal_type)

    def get_set_bits(self, goal_type):
        return [2 ** i * int(bit) for i, bit in enumerate(reversed("{0:b}".format(goal_type))) if int(bit) != 0]

    @record_time
    @profile
    def update(self):
        if self.get_blackboard_exception() is not None:
            return Status.SUCCESS
        try:
            move_cmd = self.goal.cmd_seq.pop(0)  # type: MoveCmd
            self.get_god_map().set_data(identifier.next_move_goal, move_cmd)
            cmd_id = self.get_god_map().get_data(identifier.cmd_id) + 1
            self.get_god_map().set_data(identifier.cmd_id, cmd_id)
            logging.loginfo('Planning move commands #{}/{}.'.format(cmd_id + 1, self.number_of_move_cmds))
        except IndexError:
            return Status.FAILURE

        return Status.SUCCESS

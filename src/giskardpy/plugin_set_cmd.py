from giskard_msgs.msg import MoveGoal, MoveCmd
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.exceptions import InsolvableException
from giskardpy.plugin_action_server import GetGoal


class SetCmd(GetGoal):
    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)
        self.current_goal_id = 0
        self.goal = None

    def initialise(self):
        if self.goal is None:
            self.goal = self.pop_goal()  # type: MoveGoal
            self.traj = []
            if len(self.goal.cmd_seq) == 0:
                self.raise_to_blackboard(InsolvableException(u'goal empty'))
            if self.goal.type not in [MoveGoal.PLAN_AND_EXECUTE, MoveGoal.PLAN_ONLY]:
                self.raise_to_blackboard(
                    InsolvableException(u'invalid move action goal type: {}'.format(self.goal.type)))
            self.get_god_map().safe_set_data(identifier.execute, self.goal.type == MoveGoal.PLAN_AND_EXECUTE)

    def update(self):
        # TODO goal checks should probably be its own plugin?
        if self.get_blackboard_exception():
            self.goal = None
            self.get_god_map().safe_set_data(identifier.next_move_goal, None)
            return Status.SUCCESS

        try:
            move_cmd = self.goal.cmd_seq.pop(0)  # type: MoveCmd
            self.get_god_map().safe_set_data(identifier.next_move_goal, move_cmd)
        except IndexError:
            self.goal = None
            self.get_god_map().safe_set_data(identifier.next_move_goal, None)
            return Status.SUCCESS

        return Status.RUNNING

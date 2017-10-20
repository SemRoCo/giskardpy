from collections import OrderedDict
from giskardpy.qp_problem_builder import QProblemBuilder


class Controller(object):
    def __init__(self, robot):
        self.robot = robot

        self._state = OrderedDict()  # e.g. goal
        self._soft_constraints = OrderedDict()

        self.make_constraints(self.robot)
        self.build_builder()

    def make_constraints(self, robot):
        pass

    def build_builder(self):
        self.qp_problem_builder = QProblemBuilder(self.robot.joint_constraints,
                                                  self.robot.hard_constraints,
                                                  self._soft_constraints)

    def update_observables(self, updates):
        """
        :param updates: dict{str->float} observable name to it value
        :return: dict{str->float} joint name to vel command
        """
        self._state.update(updates)

    def get_next_command(self):
        self._state.update(self.robot.get_state())
        return self.qp_problem_builder.update_observables(self._state)

    def add_frame_input(self, name):
        pass

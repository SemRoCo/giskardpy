from collections import OrderedDict
from giskardpy.qp_problem_builder import QProblemBuilder
from giskardpy.controller import Controller


class QPController(Controller):
    def __init__(self, robot, builder_backend=None):
        self.builder_backend = builder_backend
        super(QPController, self).__init__(robot)

    def init(self):
        self._controllable_constraints = OrderedDict()
        self._hard_constraints = OrderedDict()
        self._soft_constraints = OrderedDict()

        self.add_inputs(self.get_robot())
        self.make_constraints(self.get_robot())
        self.build_builder()

    def add_inputs(self, robot):
        raise (NotImplementedError)

    def make_constraints(self, robot):
        self._controllable_constraints = self.get_robot().joint_constraints
        self._hard_constraints = self.get_robot().hard_constraints

    def build_builder(self):
        self.qp_problem_builder = QProblemBuilder(self._controllable_constraints,
                                                  self._hard_constraints,
                                                  self._soft_constraints,
                                                  self.builder_backend)

    def get_next_command(self):
        # TODO add dt parameter and return next state + cmds instead of only cmds
        self.update_observables(self.get_robot().get_state())
        return self.qp_problem_builder.update_observables(self.get_state())

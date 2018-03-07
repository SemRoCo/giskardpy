import symengine_wrappers as spw
from collections import OrderedDict
from giskardpy.qp_problem_builder import QProblemBuilder, SoftConstraint
from giskardpy.robot_constraints import Robot


class Controller(object):
    def __init__(self):
        self._soft_constraints = OrderedDict()

    def init(self, *args, **kwargs):
        self.build_controller(*args, **kwargs)
        self.qp_problem_builder = QProblemBuilder(self.robot.joint_constraints,
                                                  self.robot.hard_constraints,
                                                  self._soft_constraints)

    def build_controller(self, urdf, joint_symbol_map=None, **kwargs):
        self.robot = Robot(urdf, joint_symbol_map)

    def get_cmd(self, substitutions):
        return self.qp_problem_builder.get_cmd(substitutions)

class JointController(Controller):
    def __init__(self, urdf):
        # TODO use symbol for weight
        self.default_weight = 1
        super(JointController, self).__init__()

    def default_goal_symbol_map(self):
        m = OrderedDict()
        for joint_name in self.robot.get_joint_to_symbols():
            m[joint_name] = spw.Symbol('{}_goal'.format(joint_name))
        return m

    def build_controller(self, urdf, joint_symbol_map=None, goal_symbol_map=None):
        if goal_symbol_map is None:
            goal_symbol_map = self.default_goal_symbol_map()
        for joint_name, joint_symbol in self.robot.get_joint_to_symbols().items():
            sc = SoftConstraint(lower=goal_symbol_map[joint_name] - joint_symbol,
                                upper=goal_symbol_map[joint_name] - joint_symbol,
                                weight=self.default_weight,
                                expression=joint_symbol)
            self._soft_constraints[str(joint_symbol)] = sc
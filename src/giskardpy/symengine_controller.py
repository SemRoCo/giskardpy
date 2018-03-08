import symengine_wrappers as spw
from collections import OrderedDict
from giskardpy.qp_problem_builder import QProblemBuilder, SoftConstraint
from giskardpy.robot_constraints import Robot


class Controller(object):
    def __init__(self, urdf):
        self._soft_constraints = OrderedDict()
        self.robot = Robot(urdf)

    def init(self, *args, **kwargs):
        self._build_controller(*args, **kwargs)
        controlled_joints = list(self.robot.get_joint_to_symbols().values())
        self.qp_problem_builder = QProblemBuilder(self.robot.joint_constraints,
                                                  self.robot.hard_constraints,
                                                  self._soft_constraints,
                                                  controlled_joints)

    def _build_controller(self, joint_symbol_map=None, **kwargs):
        pass

    def get_cmd(self, substitutions):
        return self.qp_problem_builder.get_cmd(substitutions)

class JointController(Controller):
    def __init__(self, urdf):
        # TODO use symbol for weight
        self.default_weight = 1
        super(JointController, self).__init__(urdf)

    def get_goal_symbol_map(self):
        return self.goal_symbol_map

    def _default_goal_symbol_map(self):
        m = OrderedDict()
        for joint_name in self.robot.get_joint_to_symbols():
            m[joint_name] = spw.Symbol('{}_goal'.format(joint_name))
        return m

    def init(self, joint_symbol_map=None, goal_symbol_map=None):
        super(JointController, self).init(joint_symbol_map=joint_symbol_map, goal_symbol_map=goal_symbol_map)

    def _build_controller(self, joint_symbol_map=None, goal_symbol_map=None):
        self.robot.set_joint_symbol_map(joint_symbol_map)
        if goal_symbol_map is None:
            self.goal_symbol_map = self._default_goal_symbol_map()
        else:
            self.goal_symbol_map = goal_symbol_map
        for joint_name, joint_symbol in self.robot.get_joint_to_symbols().items():
            sc = SoftConstraint(lower=self.goal_symbol_map[joint_name] - joint_symbol,
                                upper=self.goal_symbol_map[joint_name] - joint_symbol,
                                weight=self.default_weight,
                                expression=joint_symbol)
            self._soft_constraints[str(joint_symbol)] = sc
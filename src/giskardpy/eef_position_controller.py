from giskardpy import USE_SYMENGINE
from giskardpy.qpcontroller import QPController
from giskardpy.qp_problem_builder import SoftConstraint
from giskardpy.input_system import Point3Input, ControllerInputArray, ScalarInput

if USE_SYMENGINE:
    import giskardpy.symengine_wrappers as spw
else:
    import giskardpy.sympy_wrappers as spw

class EEFPositionControl(QPController):
    def __init__(self, robot, weight=1):
        self.weight = weight
        super(EEFPositionControl, self).__init__(robot)

    # @profile
    def add_inputs(self, robot):
        self.goal_eef = {}
        self.goal_weights = {}
        for eef in robot.end_effectors:
            self.goal_eef[eef] = Point3Input(prefix=eef, suffix='goal')
            self.goal_weights[eef] = ScalarInput(prefix=eef, suffix='sc_w')

    # @profile
    def make_constraints(self, robot):
        for eef in robot.end_effectors:
            eef_frame = robot.frames[eef]
            goal_expr = self.goal_eef[eef].get_expression()
            # dist = norm(sp.Add(pos_of(eef_frame), - goal_expr, evaluate=False))
            dist = spw.norm(spw.pos_of(eef_frame) - goal_expr)
            self._soft_constraints['align {} position'.format(eef)] = SoftConstraint(lower=-dist,
                                                                         upper=-dist,
                                                                         weight=self.goal_weights[eef].get_expression(),
                                                                         expression=dist)
            self._controllable_constraints = robot.joint_constraints
            self._hard_constraints = robot.hard_constraints
            self.update_observables({self.goal_weights[eef].get_symbol_str(): self.weight})
            self.set_goal([0,0,0])

    def set_goal(self, goal):
        """
        dict eef_name -> goal_position
        :param goal_pos:
        :return:
        """
        #for eef, goal_pos in goal.items():
        self.update_observables(self.goal_eef['gripper_link'].get_update_dict(*goal))

#!/usr/bin/env python
import unittest
from collections import OrderedDict, namedtuple
from time import time
from giskardpy.fetch import Fetch
from giskardpy.input_system import ScalarInput, FrameInput, Vec3Input, ControllerInputArray
from giskardpy.qp_problem_builder import QProblemBuilder as CythonProblemBuilder
from giskardpy.qp_problem_builder import SoftConstraint
from giskardpy.sympy_wrappers import *
import sympy as sp

PKG = 'giskardpy'

Object = namedtuple('Object', ['frame', 'dimensions', 'pclass', 'ppos', 'prot'])

class ProbabilisticObjectInput(object):
    def __init__(self, name):
        super(ProbabilisticObjectInput, self).__init__()

        self.frame = FrameInput('{}{}frame'.format(name, ControllerInputArray.separator))
        self.dimensions = Vec3Input('{}{}dimensions'.format(name, ControllerInputArray.separator))
        self.probability_class = ScalarInput('P_class')
        self.probability_pos = ScalarInput('P_trans')
        self.probability_rot   = ScalarInput('P_rot')

    def get_update_dict(self, p_object):
        out_dict = self.frame.get_update_dict(*p_object.frame)
        out_dict.update(self.dimensions.get_update_dict(p_object.dimensions))
        out_dict.update(self.probability_class.get_update_dict(p_object.pclass))
        out_dict.update(self.probability_pos.get_update_dict(p_object.ppos))
        out_dict.update(self.probability_rot.get_update_dict(p_object.prot))
        return out_dict

    def get_frame(self):
        return self.frame.get_expression()

    def get_dimensions(self):
        return self.dimensions.get_expression()

    def get_class_probability(self):
        return self.probability_class.get_expression()


class TestDiffRuntimeBug(unittest.TestCase):
    def setUp(self):
        self.obj_input = ProbabilisticObjectInput('object')
        self.robot = Fetch()
        t = time()
        self.cga = self.cylinder_grasp_affordance(self.robot.gripper, self.obj_input)
        print('cga time {}'.format(time()-t))
        self.s_dict = {'soft_constraint': SoftConstraint(-1, 1, 1, self.cga)}

    # @profile
    def cylinder_grasp_affordance(self, gripper, obj_input):
        frame = obj_input.get_frame()
        shape = obj_input.get_dimensions()
        cylinder_z = frame.col(2)
        cylinder_pos = pos_of(frame)

        gripper_x = gripper.frame.col(0)
        gripper_z = gripper.frame.col(2)
        gripper_pos = pos_of(gripper.frame)
        c_to_g = sp.Add(gripper_pos, - cylinder_pos, evaluate=False)

        zz_align = sp.Abs(gripper_z.dot(cylinder_z), evaluate=False)
        xz_align = gripper_x.dot(cylinder_z)
        dist_z = cylinder_z.dot(c_to_g)
        border_z = (shape[2] - gripper.height) * 0.5
        cap_dist_normalized_signed = dist_z / border_z
        cap_dist_normalized = sp.Abs(cap_dist_normalized_signed, evaluate=False)

        cap_top_grasp = 1 - sp.Max(-xz_align * sp.Min(cap_dist_normalized_signed, 1), 0, evaluate=False)
        cap_bottom_grasp = 1 - sp.Min(xz_align * sp.Max(cap_dist_normalized_signed, -1), 0, evaluate=False)

        dist_z_center_normalized = sp.Max(1 - cap_dist_normalized, 0, evaluate=False)
        dist_ax = sp.sqrt(frame.col(0).dot(c_to_g) ** 2 + frame.col(1).dot(c_to_g) ** 2)

        center_grasp = (1 - dist_z_center_normalized - dist_ax) * zz_align

        return sp.Max(center_grasp, cap_top_grasp, cap_bottom_grasp) * obj_input.get_class_probability()

    def test_differentiation_speed_cython(self):
        cy_builder = CythonProblemBuilder(self.robot.joint_constraints, self.robot.hard_constraints, self.s_dict)


if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestDiffRuntimeBug',
                    test=TestDiffRuntimeBug)
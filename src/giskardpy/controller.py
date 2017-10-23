from collections import OrderedDict
from giskardpy.qp_problem_builder import QProblemBuilder
from giskardpy.robot import Robot
from giskardpy.sympy_wrappers import *

class ControllerInput(object):
    separator = '__'

    def __init__(self, name):
        super(ControllerInput, self).__init__()
        self.name = name
        self.expression = None
        self.observables = []

    def get_value(self, obs_dict):
        return self.expression.subs(obs_dict)

    def update_observables(self, obs_dict, obs):
        pass

    def get_observables(self):
        return self.observables


class ScalarInput(ControllerInput):
    def __init__(self, name):
        super(ScalarInput, self).__init__(name)
        self.expression = sp.Symbol(name)
        self.observables.append(self.expression)

    def update_observables(self, obs_dict, obs):
        obs_dict[self.name] = obs


class Vec3Input(ControllerInput):
    def __init__(self, name):
        super(Vec3Input, self).__init__(name)
        self.xn = name + ControllerInput.separator + 'x'
        self.yn = name + ControllerInput.separator + 'y'
        self.zn = name + ControllerInput.separator + 'z'
        self.x = sp.Symbol(self.xn)
        self.y = sp.Symbol(self.yn)
        self.z = sp.Symbol(self.zn)
        self.observables += [self.x, self.y, self.z]
        self.expression = vec3(self.x, self.y, self.z)

    def update_observables(self, obs_dict, x, y, z):
        obs_dict[self.xn] = x
        obs_dict[self.yn] = y
        obs_dict[self.zn] = z


class Point3Input(Vec3Input):
    def __init__(self, name):
        super(Point3Input, self).__init__(name)
        self.expression = point3(self.x, self.y, self.z)

class RPYInput(ControllerInput):
    def __init__(self, name):
        super(RPYInput, self).__init__(name)
        self.rn = name + ControllerInput.separator + 'r'
        self.pn = name + ControllerInput.separator + 'p'
        self.yn = name + ControllerInput.separator + 'y'
        self.r = sp.Symbol(self.rn)
        self.p = sp.Symbol(self.pn)
        self.y = sp.Symbol(self.yn)
        self.observables += [self.r, self.p, self.y]
        self.expression += rotation3_rpy(self.r, self.p, self.y)

    def update_observables(self, obs_dict, r, p, y):
        obs_dict[self.rn] = r
        obs_dict[self.pn] = p
        obs_dict[self.yn] = y


class QuaternionInput(ControllerInput):
    def __init__(self, name):
        super(RPYInput, self).__init__(name)
        self.xn = name + ControllerInput.separator + 'x'
        self.yn = name + ControllerInput.separator + 'y'
        self.zn = name + ControllerInput.separator + 'z'
        self.wn = name + ControllerInput.separator + 'w'
        self.x = sp.Symbol(self.xn)
        self.y = sp.Symbol(self.yn)
        self.z = sp.Symbol(self.zn)
        self.w = sp.Symbol(self.wn)
        self.observables += [self.x, self.y, self.z, self.w]
        self.expression += rotation3_quat(self.x, self.y, self.z, self.w)

    def update_observables(self, obs_dict, x, y, z, w):
        obs_dict[self.xn] = x
        obs_dict[self.yn] = y
        obs_dict[self.zn] = z
        obs_dict[self.wn] = w


class FrameInput(ControllerInput):
    def __init__(self, name):
        super(FrameInput, self).__init__(name)
        self.rotInput = QuaternionInput(name + ControllerInput.separator + 'rot')
        self.locInput = Point3Input(name + ControllerInput.separator + 'loc')

    def update_observables(self, obs_dict, q1, q2, q3, q4, x, y, z):
        self.rotInput.update_observables(obs_dict, q1, q2, q3, q4)
        self.locInput.update_observables(obs_dict, x, y, z)

    def get_observables(self):
        return self.rotInput.get_observables() + self.locInput.get_observables()


class Controller(object):
    def __init__(self, robot):
        self.robot = robot

        #TODO: fill in child class
        self._observables = []
        self.soft_constraints = OrderedDict()
        self.controllable_constraints = OrderedDict()
        self.inputs = {}

        self._state = OrderedDict()  # e.g. goal
        self._soft_constraints = OrderedDict()

        self.make_constraints(self.robot)
        self.build_builder()

    def make_constraints(self, robot):
        pass

    def build_builder(self):
        self.make_constraints(self.robot)
        self.qp_problem_builder = QProblemBuilder(self.controllable_constraints,
                                                  self.robot.hard_constraints,
                                                  self._soft_constraints)

    def update_observables(self, updates):
        """
        :param updates: dict{str->float} observable name to it value
        :return: dict{str->float} joint name to vel command
        """
        if updates is None:
            updates = {}
        self._state.update(updates)
        # return self.qp_problem_builder.update_observables(updates)


    def get_next_command(self):
        self._state.update(self.robot.get_state())
        return self.qp_problem_builder.update_observables(self._state)


    def get_observables(self):
        return self.get_robot_observables() + self.get_controller_observables()

    def get_controller_observables(self):
        return self._observables

    def get_robot_observables(self):
        return self.robot.observables

    def add_input(self, name, cls):
        if name in self.inputs:
            out = self.inputs[name]
            if isinstance(out, cls):
                return out
            else:
                raise Exception("Can't add input '{}' of type '{}' as that name is already taken by an input of type '{}'.".format(name, str(cls), str(type(out))))
        else:
            out = cls(name)
            self.inputs[name] = out
            return out

    def add_scalar_input(self, name):
        return self.add_input(name, ScalarInput)

    def add_vec3_input(self, name):
        return self.add_input(name, Vec3Input)

    def add_point3_input(self, name):
        return self.add_input(name, Point3Input)

    def add_frame_input(self, name):
        return self.add_input(name, FrameInput)

    def update_input(self, name, *args):
        if not name in self.inputs:
            raise Exception('Input "{}" does not exist.'.format(name))
        self.inputs[name].update_observables(self.state, *args)


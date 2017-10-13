from giskardpy.qp_problem_builder import QPProblemBuilder
from giskardpy.sympy_wrappers import vec3
from sympy import *
import numpy as np

from qpoases import PySQProblem as SQProblem
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel

class ControllableConstraint(object):
    def __init__(self, lower, upper, weight, joint):
        self.lower  = lower
        self.upper  = upper
        self.weight = weight
        self.joint  = joint


class SoftConstraint(object):
    def __init__(self, lower, upper, weight, expression, name):
        self.lower  = lower
        self.upper  = upper
        self.weight = weight
        self.expression = expression
        self.name   = name


class HardConstraint(object):
    def __init__(self, lower, upper, expression, name=''):
        self.lower = lower
        self.upper = upper
        self.expression = expression
        self.name  = name


class ControllerInput(object):
    def __init__(self, name, controller):
        if ':' in name:
            raise Exception("Inputs may not contain ':'")
        self.name = name
        self.controller = controller
        self.expression = None


class JointInput(ControllerInput):
    def __init__(self, name, controller, expression):
        super(JointInput, self).__init__(name, controller)
        self.expression = Symbol(name)
        self.setValue(0)

    def setValue(self, value):
        if not isinstance(value, int) and not isinstance(value, float):
            raise Exception(
                "Can not set the value of joint input '" + self.name + "' using an object of type '" + str(type(value)) + "'")

        self.controller.state[self.name] = value


class ScalarInput(ControllerInput):
    def __init__(self, name, controller, expression):
        super(ScalarInput, self).__init__(name, controller)
        self.expression = Symbol(name)
        self.setValue(0)

    def setValue(self, value):
        if not isinstance(value, int) and not isinstance(value, float):
            raise Exception("Can not set the value of scalar input '" + self.name + "' using an object of type '" + str(type(value)) + "'")

        self.controller.state[self.name] = value


class VectorInput(ControllerInput):
    def __init__(self, name, controller, expression):
        super(VectorInput, self).__init__(name, controller)
        self.expression = vec3(Symbol(name + ':X'), Symbol(name + ':Y'), Symbol(name + ':Z'))
        self.setValue([0,0,0])

    def setValue(self, value):
        if not isinstance(value, list):
            raise Exception("Can't set value of vector input '" + self.name + "' using a value of type '" + str(type(value)) + "'")
        if len(value) != 3:
            raise Exception("List does not have a length of 3")
        for n in range(len(value)):
            x = value[n]
            if not isinstance(x, int) and not isinstance(x, float):
                raise Exception("Value at index " + str(n) + " of type '" + str(type(x)) + "' which can not be used to as vector component.")

        self.controller.state[self.name + ':X'] = value
        self.controller.state[self.name + ':Y'] = value
        self.controller.state[self.name + ':Z'] = value


class QPController(object):
    def __init__(self):
        self.inputs = {}
        self.state  = {}
        self.qpBuilder = None
        self.initialized = False
        self.qpProblem = None
        self.xdot_full = None
        self.xdot_control = None
        self.xdot_slack = None


    def scalarInput(self, name):
        """Creates a scalar input with the given name and returns the matching SymPy expression.
           If a scalar input with the same name already exists, its expression is returned.
           If the name is already taken by an exception of another type an exception is raised.
        """
        if name in self.inputs:
            input = self.inputs[name]
            if isinstance(input, ScalarInput):
                return input.expression
            else:
                raise Exception("Can't a scalar input with name '" + name + " as an input of type " + str(type(input)) + " with that name already exists.")

        else:
            input = ScalarInput(name, self)
            self.inputs[name] = input
            return input.expression

    def jointInput(self, name):
        """Creates a joint input with the given name and returns the matching SymPy expression.
           If a joint input with the same name already exists, its expression is returned.
           If the name is already taken by an exception of another type an exception is raised.
        """
        if name in self.inputs:
            input = self.inputs[name]
            if isinstance(input, JointInput):
                return input.expression
            else:
                raise Exception("Can't a joint input with name '" + name + " as an input of type " + str(type(input)) + " with that name already exists.")

        else:
            input = JointInput(name, self)
            self.inputs[name] = input
            return input.expression

    def vectorInput(self, name):
        """Creates a vector input with the given name and returns the matching SymPy expression.
           If a vector input with the same name already exists, its expression is returned.
           If the name is already taken by an exception of another type an exception is raised.
        """
        if name in self.inputs:
            input = self.inputs[name]
            if isinstance(input, VectorInput):
                return input.expression
            else:
                raise Exception("Can't a scalar input with name '" + name + " as an input of type " + str(
                    type(input)) + " with that name already exists.")

        else:
            input = VectorInput(name, self)
            self.inputs[name] = input
            return input.expression

    def setInput(self, name, value):
        if not name in self.inputs:
            raise Exception("Unknown input '" + name + "'")

        self.inputs[name].setValue(value)

    def initialize(self, controllables, softConstraints, hardConstraints):
        cont_lower  = []
        cont_upper  = []
        cont_weight = []
        cont_names  = []

        soft_lower      = []
        soft_upper      = []
        soft_weight     = []
        soft_expression = []

        hard_lower      = []
        hard_upper      = []
        hard_expression = []

        symbols = set()

        for c in controllables:
            if c.name in cont_names:
                raise Exception("Redefinition of controllable constraint for joint '" + c.name + "'")
            symbols = symbols.union(c.lower.free_symbols).union(c.upper.free_symbols).union(c.weight.free_symbols)
            cont_lower.append(c.lower)
            cont_upper.append(c.upper)
            cont_weight.append(c.weight)
            cont_names.append(c.name)

        for s in softConstraints:
            symbols = symbols.union(s.lower).union(s.upper).union(s.weight).union(s.expression)
            soft_lower.append(s.lower)
            soft_upper.append(s.upper)
            soft_weight.append(s.weight)
            soft_expression.append(s.expression)

        for h in hardConstraints:
            symbols = symbols.union(h.lower).union(h.upper).union(h.expression)
            hard_lower.append(h.lower)
            hard_upper.append(h.upper)
            hard_expression.append(h.expression)

        for s in symbols:
            if s not in self.state:
                raise Exception("Symbol '" + str(s) + "' is a part of the controller's expressions but not of its state.")


        self.qpBuilder = QPProblemBuilder(cont_lower, cont_upper, cont_weight,
                                          soft_lower, soft_upper, soft_weight, soft_expression,
                                          hard_lower, hard_upper, hard_expression, symbols, cont_names)
        self.controllable_names = cont_names
        self.initialized = True


    def start(self, nWSR):
        if not self.initialized:
            raise Exception("Tried to start controller without initialization.")

        self.qpBuilder.update(self.state)
        self.qpProblem = SQProblem(self.qpBuilder.num_weights(), self.qpBuilder.num_constraints())
        options = Options()
        options.printLevel = PrintLevel.NONE
        self.qpProblem.setOptions(options)
        success = self.qpProblem.init(self.qpBuilder.np_H,  self.qpBuilder.np_g, self.qpBuilder.np_A,
                                      self.qpBuilder.np_lb, self.qpBuilder.np_ub,
                                      self.qpBuilder.np_lbA,self.qpBuilder.np_ubA, nWSR)

        if success != qpoases.SUCCESSFUL_RETURN:
            print("Failed to initialize QP-problem. ERROR:")
            print(qpoases.MessageHandling.getErrorCodeMessage(success))
            self.qpBuilder.printInternals()
            print("nWSR: " + str(nWSR))
            return False

        self.xdot_full    = np.zeros(self.qpBuilder.num_weights())
        self.xdot_control = np.zeros(self.qpBuilder.num_controllables())
        self.xdot_slack   = np.zeros(self.qpBuilder.num_soft_constraints())

        return True


    def update(self, nWSR):
        if not self.initialized:
            raise Exception("Attempted to update controller without initialization.")
        if self.xdot_control == None:
            raise Exception("Attempted to update controller without starting it first.")

        self.qpBuilder.update(self.state)

        success = self.qpProblem.hotstart(self.qpBuilder.np_H,  self.qpBuilder.np_g, self.qpBuilder.np_A,
                                          self.qpBuilder.np_lb, self.qpBuilder.np_ub,
                                          self.qpBuilder.np_lbA,self.qpBuilder.np_ubA, nWSR)

        if success != qpoases.SUCCESSFUL_RETURN:
            return False

        self.qpProblem.getPrimalSolution(self.xdot_full)
        self.xdot_control = self.xdot_full[:self.qpBuilder.num_controllables()]
        self.xdot_slack = self.xdot_full[self.qpBuilder.num_controllables():][:self.qpBuilder.num_soft_constraints()]

        return True
    

    def getCommandVector(self):
        return self.xdot_control


    def getCommand(self):
        if self.xdot_control is None:
            raise Exception("Can not give commands without the controller running first")
        if len(self.xdot_control) != len(self.controllable_names):
            raise Exception("Command vector and list of controllable names are not of equal length. Something went very wrong...")

        out = {}
        for x in range(len(self.xdot_control)):
            out[self.controllable_names[x]] = self.xdot_control[x]

        return out





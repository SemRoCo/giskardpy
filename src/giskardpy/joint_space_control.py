from giskardpy.contraint_templates import minimize_array
from giskardpy.controller import Controller


class JointSpaceControl(Controller):
    def __init__(self, robot, weight=1):
        self.weight = weight
        self.joint_to_goal = {}
        super(JointSpaceControl, self).__init__(robot)

    def make_constraints(self, robot):
        for joint, goal, weight, sc in zip(robot.get_joint_names(), *minimize_array(robot.get_joint_names())):
            self._soft_constraints[goal] = sc
            self._state[weight] = self.weight
            self.joint_to_goal[joint] = goal

    def set_goal(self, goal_dict):
        self.update_observables({self.joint_to_goal[k]: v for k,v in goal_dict.items()})

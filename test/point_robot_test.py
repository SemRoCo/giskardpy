from giskardpy.qp_controller import *
from giskardpy.sympy_wrappers import vec3

class PointRobot(object):
    """This is a simple point robot. It's workspace is limited to 6*6*6m around its origin"""
    def __init__(self, controller):
        super(PointRobot, self).__init__()

        self.end_effector = vec3(controller.jointInput('joint_x'), controller.jointInput('joint_y'), controller.jointInput('joint_z'))
        self.controllables = [ControllableConstraint(-0.2, 0.2, 0.01, 'joint_x'),
                              ControllableConstraint(-0.2, 0.2, 0.01, 'joint_y'),
                              ControllableConstraint(-0.2, 0.2, 0.01, 'joint_z')]
        self.hard_constraints = [HardConstraint(-3 - self.end_effector.x, 3 - self.end_effector.x, self.end_effector.x),
                                 HardConstraint(-3 - self.end_effector.y, 3 - self.end_effector.y, self.end_effector.y),
                                 HardConstraint(-3 - self.end_effector.z, 3 - self.end_effector.z, self.end_effector.z)]

        self.command_topic = '/point_bot/velocity_commands'


def moveToSC(pointA, pointB, weight=1):
    """This function generates a soft constraint tries to
    drive the distance between two points down to 0
    """
    dist = (pointB - pointA).magnitude()
    return SoftConstraint(-dist, -dist, weight, dist, 'Align points constraint')

if __name__ == '__main__':

    controller = QPController()
    robot = PointRobot(controller)

    goal = controller.vectorInput('goal')

    controller.initialize(robot.controllables, [moveToSC(robot.end_effector, goal)], robot.hard_constraints)

    controller.setInput('goal', [1,1,-1])
    controller.setInput('joint_x', 0)
    controller.setInput('joint_y', -1)
    controller.setInput('joint_z', 0.5)

    controller.start(1000)
    controller.update(1000)

    command = controller.getCommand()

    print(str(command))
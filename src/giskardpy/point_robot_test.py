from giskardpy.qp_controller import *
from giskardpy.sympy_wrappers import *

from sympy import init_printing
init_printing(pretty_print=True)

class PointRobot(object):
    """This is a simple point robot. It's workspace is limited to 6*6*6m around its origin"""
    def __init__(self, controller):
        super(PointRobot, self).__init__()

        self.end_effector = Vector(controller.jointInput('joint_x'), controller.jointInput('joint_y'), controller.jointInput('joint_z'))
        self.controllables = [ControllableConstraint(-0.2, 0.2, 0.01, 'joint_x'),
                              ControllableConstraint(-0.2, 0.2, 0.01, 'joint_y'),
                              ControllableConstraint(-0.2, 0.2, 0.01, 'joint_z')]
        self.hard_constraints = [HardConstraint(-3 - self.end_effector[0], 3 - self.end_effector[0], self.end_effector[0]),
                                 HardConstraint(-3 - self.end_effector[1], 3 - self.end_effector[1], self.end_effector[1]),
                                 HardConstraint(-3 - self.end_effector[2], 3 - self.end_effector[2], self.end_effector[2])]

        self.command_topic = '/point_bot/velocity_commands'


def moveToSC(pointA, pointB, weight=1):
    """This function generates a soft constraint tries to
    drive the distance between two points down to 0
    """
    return [SoftConstraint(pointB[0] - pointA[0], pointB[0] - pointA[0], weight, pointA[0], 'Align points x'),
            SoftConstraint(pointB[1] - pointA[1], pointB[1] - pointA[1], weight, pointA[1], 'Align points y'),
            SoftConstraint(pointB[2] - pointA[2], pointB[2] - pointA[2], weight, pointA[2], 'Align points z')]

if __name__ == '__main__':

    controller = QPController()
    robot = PointRobot(controller)

    goal = controller.vectorInput('goal')

    controller.initialize(robot.controllables, moveToSC(robot.end_effector, goal), robot.hard_constraints)
    controller.qpBuilder.printInternals()


    jointX = 0
    jointY = -1
    jointZ = 0.5

    controller.setInput('goal', [1,1,-1])
    controller.setInput('joint_x', jointX)
    controller.setInput('joint_y', jointY)
    controller.setInput('joint_z', jointZ)

    nWSR = np.array([1000])

    try:
        controller.start(nWSR)
    except Exception as e:
        print('An error occurred during controller start: ' + str(e))
        controller.printState()
        controller.qpBuilder.printInternals()
        exit(-1)

    # simulate 12 seconds of movement
    for x in range(600):
        controller.update(nWSR)

        command = controller.getCommand()
        jointX += command['joint_x'] * 0.02
        jointY += command['joint_y'] * 0.02
        jointZ += command['joint_z'] * 0.02
        controller.setInput('joint_x', jointX)
        controller.setInput('joint_y', jointY)
        controller.setInput('joint_z', jointZ)

    print('last command: ' + str(controller.getCommand()))
    print('joint_x: ' + str(jointX))
    print('joint_y: ' + str(jointY))
    print('joint_z: ' + str(jointZ))
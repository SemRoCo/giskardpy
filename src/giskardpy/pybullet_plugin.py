import rospkg
import pybullet as p
from collections import OrderedDict
from time import time

import pybullet_data
import rospy

from giskardpy.plugin import IOPlugin

def replace_paths(urdf_str):
    rospack = rospkg.RosPack()
    with open('/tmp/robot.urdf', 'w') as o:
        for line in urdf_str.split('\n'):
            if 'package://' in line:
                package_name = line.split('package://', 1)[-1].split('/',1)[0]
                real_path = rospack.get_path(package_name)
                o.write(line.replace(package_name, real_path))
            else:
                o.write(line)

class PyBullet(IOPlugin):
    def get_readings(self):
        # print(p.getClosestPoints(self.robot_bullet_id, self.robot_bullet_id, 0.05))
        return super(PyBullet, self).get_readings()

    def update(self, databus):
        js = databus.get_data(self.js_identifier)
        for i, joint_name in enumerate(js.name):
            p.resetJointState(self.robot_bullet_id, self.joint_map[joint_name], js.position[i])

    def start(self):
        physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.8)
        planeId = p.loadURDF('plane.urdf')
        cubeStartPos = [0, 0, 0]
        cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_bullet_id = p.loadURDF('/tmp/robot.urdf', cubeStartPos, cubeStartOrientation,
                                          flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)

        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self.init_js_info()

    def stop(self):
        p.disconnect()

    def __init__(self):
        self.js_identifier = 'js'
        self.robot_bullet_id = 0
        #TODO get robot description from databus
        urdf = rospy.get_param('robot_description')
        t = time()
        replace_paths(urdf)
        super(PyBullet, self).__init__()

    def init_js_info(self):
        self.joint_map = OrderedDict()
        for joint_index in range(p.getNumJoints(self.robot_bullet_id)):
            joint_info = p.getJointInfo(self.robot_bullet_id, joint_index)
            joint_name = joint_info[1]
            self.joint_map[joint_name] = joint_index
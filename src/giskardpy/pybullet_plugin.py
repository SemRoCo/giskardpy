import rospy

from giskardpy.plugin import IOPlugin
from giskardpy.pybullet_world import PyBulletWorld


class PyBullet(IOPlugin):
    def get_readings(self):
        print(self.world.check_collision())
        return super(PyBullet, self).get_readings()

    def update(self, databus):
        js = databus.get_data(self.js_identifier)
        self.world.set_joint_state(self.robot_name, js)

    def start(self):
        self.world.activate_viewer()
        #TODO get robot description from databus
        urdf = rospy.get_param('robot_description')
        self.world.spawn_urdf_robot(urdf, self.robot_name)

    def stop(self):
        self.world.deactivate_viewer()

    def __init__(self):
        self.js_identifier = 'js'
        self.robot_name = 'pr2'
        self.world = PyBulletWorld()
        super(PyBullet, self).__init__()

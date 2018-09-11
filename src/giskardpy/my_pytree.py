from Queue import Queue, Empty
from time import time

import rospy
from py_trees import Behaviour, Blackboard, Status, Sequence, BehaviourTree
from sensor_msgs.msg import JointState

from giskardpy.god_map import GodMap
from giskardpy.plugin_joint_state import JSBehavior
from giskardpy.plugin_pybullet import InitPyBulletWorldB, PyBulletMonitorB
from giskardpy.utils import to_joint_state_dict


class PrintJs(Behaviour):

    def __init__(self, name, *args, **kwargs):
        super(PrintJs, self).__init__(name, *args, **kwargs)

    def setup(self, timeout):
        b = Blackboard()
        self.god_map = b.god_map
        return super(PrintJs, self).setup(timeout)

    def update(self):
        print(self.god_map.safe_get_data(['js']))
        return Status.SUCCESS


if __name__ == '__main__':
    rospy.init_node('smach_test')
    prefix = 'giskard'

    root_tips = rospy.get_param(u'{}/interactive_marker_chains'.format(prefix))
    # gui = rospy.get_param(u'~enable_gui')
    gui = True
    map_frame = rospy.get_param(u'{}/map_frame'.format(prefix))
    joint_convergence_threshold = rospy.get_param(u'{}/joint_convergence_threshold'.format(prefix))
    wiggle_precision_threshold = rospy.get_param(u'{}/wiggle_precision_threshold'.format(prefix))
    sample_period = rospy.get_param(u'{}/sample_period'.format(prefix))
    default_joint_vel_limit = rospy.get_param(u'{}/default_joint_vel_limit'.format(prefix))
    default_collision_avoidance_distance = rospy.get_param(u'{}/default_collision_avoidance_distance'.format(prefix))
    fill_velocity_values = rospy.get_param(u'{}/fill_velocity_values'.format(prefix))
    nWSR = rospy.get_param(u'{}/nWSR'.format(prefix))
    root_link = rospy.get_param(u'{}/root_link'.format(prefix))
    marker = rospy.get_param(u'{}/enable_collision_marker'.format(prefix))
    enable_self_collision = rospy.get_param(u'{}/enable_self_collision'.format(prefix))
    if nWSR == u'None':
        nWSR = None
    path_to_data_folder = rospy.get_param(u'{}/path_to_data_folder'.format(prefix))
    collision_time_threshold = rospy.get_param(u'{}/collision_time_threshold'.format(prefix))
    max_traj_length = rospy.get_param(u'{}/max_traj_length'.format(prefix))
    # path_to_data_folder = '/home/ichumuh/giskardpy_ws/src/giskardpy/data/pr2'
    if not path_to_data_folder.endswith(u'/'):
        path_to_data_folder += u'/'

    fk_identifier = u'fk'
    cartesian_goal_identifier = u'goal'
    js_identifier = u'js'
    controlled_joints_identifier = u'controlled_joints'
    trajectory_identifier = u'traj'
    time_identifier = u'time'
    next_cmd_identifier = u'motor'
    collision_identifier = u'collision'
    closest_point_identifier = u'cpi'
    collision_goal_identifier = u'collision_goal'
    pyfunction_identifier = u'pyfunctions'
    controllable_links_identifier = u'controllable_links'
    robot_description_identifier = u'robot_description'
    pybullet_identifier = u'pybullet_world'

    blackboard = Blackboard()
    blackboard.god_map = GodMap()


    root = Sequence('root')

    init = Sequence('init')
    init.add_child(InitPyBulletWorldB('initpb', pybullet_identifier, path_to_data_folder, gui))

    monitor = Sequence('monitor')
    monitor.add_child(JSBehavior('JS', js_identifier))
    monitor.add_child(PyBulletMonitorB('pbm', js_identifier, pybullet_identifier, map_frame, root_link))

    root.add_child(init)
    root.add_child(monitor)

    tree = BehaviourTree(root)

    def pre_tick(bt):
        blackboard.time = time()

    def post_tick(bt):
        if bt.count==10000:
            print(time() - blackboard.time)

    # tree.add_pre_tick_handler(pre_tick)
    tree.add_post_tick_handler(post_tick)

    tree.setup(30)
    blackboard.time = time()
    tree.tick_tock(0)

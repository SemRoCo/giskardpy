from time import time

import smach
import rospy
from sensor_msgs.msg import JointState

from giskardpy.god_map import GodMap
from smach import State, StateMachine, set_loggers

from giskardpy.plugin import SleepState
from giskardpy.plugin_joint_state import MonitorJS
from giskardpy.plugin_pybullet import PybulletMonitorState, InitPyBulletWorld
from giskardpy.utils import to_joint_state_dict



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


    def do_nothing(*args):
        pass
    set_loggers(info=do_nothing, warn=smach.logwarn, debug=smach.logdebug, error=smach.logerr)

    top = StateMachine(outcomes=['end'])
    top.userdata.god_map = GodMap()

    with top:
        init = StateMachine(outcomes=['end'], input_keys=['god_map'], output_keys=['god_map'])
        with init:
            StateMachine.add('init bullet', InitPyBulletWorld(pybullet_identifier,
                                                              path_to_data_folder,
                                                              gui),
                             transitions={InitPyBulletWorld.Finished: 'end'})

        monitor = StateMachine(outcomes=['end'], input_keys=['god_map'], output_keys=['god_map'])
        with monitor:
            StateMachine.add('js', MonitorJS(js_identifier),
                             transitions={MonitorJS.Finished: 'p'})
            StateMachine.add('p', PybulletMonitorState(js_identifier, pybullet_identifier,
                                                       map_frame, root_link, path_to_data_folder,
                                                       gui),
                             transitions={PybulletMonitorState.Finished: 'sleep'})
            StateMachine.add('sleep', SleepState(),
                             transitions={SleepState.Finished: 'js'})

        StateMachine.add('init', init, transitions={'end': 'monitor'})
        StateMachine.add('monitor', monitor, transitions={'end': 'monitor'})

    top.userdata.god_map.set_data(['c'],0)
    top.userdata.god_map.set_data(['time'], time())
    print(top.execute(parent_ud=None))
import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion

import giskardpy_ros.ros1.tfwrapper as tf
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware, set_middleware
from giskardpy.model.joints import OneDofJoint
from giskardpy.utils.math import quaternion_from_axis_angle
from giskardpy_ros.ros1.interface import ROS1Wrapper
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard
from giskardpy_ros.utils.utils_for_tests import launch_launchfile

from giskardpy_ros.utils.utils_for_tests import GiskardTestWrapper


@pytest.fixture(scope='module')
def ros(request):
    get_middleware().loginfo('init ros')
    rospy.init_node('tests')
    set_middleware(ROS1Wrapper())
    tf.init(60)

    def kill_ros():
        try:
            GiskardBlackboard().tree.render()
        except KeyError as e:
            get_middleware().logerr(f'Failed to render behavior tree.')
        get_middleware().loginfo('shutdown ros')
        rospy.signal_shutdown('die')

    try:
        rospy.get_param('kitchen_description')
    except:
        try:
            launch_launchfile('package://iai_kitchen/launch/upload_kitchen_obj.launch')
        except:
            get_middleware().logwarn('iai_apartment not found')
    try:
        rospy.get_param('apartment_description')
    except:
        try:
            launch_launchfile('package://iai_apartment/launch/upload_apartment.launch')
        except:
            get_middleware().logwarn('iai_kitchen not found')
    request.addfinalizer(kill_ros)


@pytest.fixture()
def resetted_giskard(giskard: GiskardTestWrapper) -> GiskardTestWrapper:
    get_middleware().loginfo('resetting giskard')
    giskard.restart_ticking()
    giskard.clear_motion_goals_and_monitors()
    if GiskardBlackboard().tree.is_standalone() and giskard.has_odometry_joint():
        zero = PoseStamped()
        zero.header.frame_id = 'map'
        zero.pose.orientation.w = 1
        done = giskard.monitors.add_set_seed_odometry(zero, name='initial pose')
        giskard.allow_all_collisions()
        giskard.monitors.add_end_motion(start_condition=done)
        giskard.execute(add_local_minimum_reached=False)
    giskard.world.clear()
    giskard.reset()
    return giskard


@pytest.fixture()
def zero_pose(resetted_giskard: GiskardTestWrapper) -> GiskardTestWrapper:
    if GiskardBlackboard().tree.is_standalone():
        done = resetted_giskard.monitors.add_set_seed_configuration(resetted_giskard.default_pose,
                                                                    name='initial joint state')
        resetted_giskard.allow_all_collisions()
        resetted_giskard.monitors.add_end_motion(start_condition=done)
        resetted_giskard.execute(add_local_minimum_reached=False)
    else:
        resetted_giskard.allow_all_collisions()
        done = resetted_giskard.motion_goals.add_joint_position(name='joint goal', goal_state=resetted_giskard.default_pose)
        resetted_giskard.monitors.add_end_motion(start_condition=done)
        resetted_giskard.execute(add_local_minimum_reached=False)
    return resetted_giskard


@pytest.fixture()
def better_pose(resetted_giskard: GiskardTestWrapper) -> GiskardTestWrapper:
    if GiskardBlackboard().tree.is_standalone():
        done = resetted_giskard.monitors.add_set_seed_configuration(resetted_giskard.better_pose,
                                                                    name='initial joint state')
        resetted_giskard.allow_all_collisions()
        resetted_giskard.monitors.add_end_motion(start_condition=done)
        resetted_giskard.execute(add_local_minimum_reached=False)
    else:
        resetted_giskard.allow_all_collisions()
        resetted_giskard.set_joint_goal(resetted_giskard.better_pose)
        resetted_giskard.execute()
    return resetted_giskard


@pytest.fixture()
def kitchen_setup(better_pose: GiskardTestWrapper) -> GiskardTestWrapper:
    better_pose.default_env_name = 'iai_kitchen'
    if GiskardBlackboard().tree.is_standalone():
        kitchen_pose = PoseStamped()
        kitchen_pose.header.frame_id = str(better_pose.default_root)
        kitchen_pose.pose.orientation.w = 1
        better_pose.add_urdf_to_world(name=better_pose.default_env_name,
                                      urdf=rospy.get_param('kitchen_description'),
                                      pose=kitchen_pose)
    else:
        kitchen_pose = tf.lookup_pose('map', 'iai_kitchen/world')
        better_pose.add_urdf_to_world(name=better_pose.default_env_name,
                                      urdf=rospy.get_param('kitchen_description'),
                                      pose=kitchen_pose,
                                      js_topic='/kitchen/joint_states',
                                      set_js_topic='/kitchen/cram_joint_states')
    js = {}
    for joint_name in god_map.world.groups[better_pose.default_env_name].movable_joint_names:
        joint = god_map.world.joints[joint_name]
        if isinstance(joint, OneDofJoint):
            if GiskardBlackboard().tree.is_standalone():
                js[str(joint.free_variable.name)] = 0.0
            else:
                js[str(joint.free_variable.name.short_name)] = 0.0
    better_pose.set_env_state(js)
    return better_pose

@pytest.fixture()
def dlr_kitchen_setup(better_pose: GiskardTestWrapper) -> GiskardTestWrapper:
    better_pose.default_env_name = 'dlr_kitchen'
    if GiskardBlackboard().tree.is_standalone():
        kitchen_pose = PoseStamped()
        kitchen_pose.header.frame_id = str(better_pose.default_root)
        kitchen_pose.pose.position.x = -2
        kitchen_pose.pose.position.y = 2
        kitchen_pose.pose.orientation = Quaternion(*quaternion_from_axis_angle([0,0,1], -np.pi/2))
        better_pose.add_urdf_to_world(name=better_pose.default_env_name,
                                      urdf=rospy.get_param('kitchen_description'),
                                      pose=kitchen_pose)
    else:
        kitchen_pose = tf.lookup_pose('map', 'iai_kitchen/world')
        better_pose.add_urdf_to_world(name=better_pose.default_env_name,
                                      urdf=rospy.get_param('kitchen_description'),
                                      pose=kitchen_pose,
                                      js_topic='/kitchen/joint_states',
                                      set_js_topic='/kitchen/cram_joint_states')
    js = {}
    for joint_name in god_map.world.groups[better_pose.default_env_name].movable_joint_names:
        joint = god_map.world.joints[joint_name]
        if isinstance(joint, OneDofJoint):
            if GiskardBlackboard().tree.is_standalone():
                js[str(joint.free_variable.name)] = 0.0
            else:
                js[str(joint.free_variable.name.short_name)] = 0.0
    better_pose.set_env_state(js)
    return better_pose


@pytest.fixture()
def apartment_setup(better_pose: GiskardTestWrapper) -> GiskardTestWrapper:
    better_pose.default_env_name = 'iai_apartment'
    if GiskardBlackboard().tree.is_standalone():
        kitchen_pose = PoseStamped()
        kitchen_pose.header.frame_id = str(better_pose.default_root)
        kitchen_pose.pose.orientation.w = 1
        better_pose.add_urdf_to_world(name=better_pose.default_env_name,
                                      urdf=rospy.get_param('apartment_description'),
                                      pose=kitchen_pose)
    else:
        better_pose.add_urdf_to_world(name=better_pose.default_env_name,
                                      urdf=rospy.get_param('apartment_description'),
                                      pose=tf.lookup_pose('map', 'iai_apartment/apartment_root'),
                                      js_topic='/apartment_joint_states',
                                      set_js_topic='/iai_kitchen/cram_joint_states')
    js = {}
    for joint_name in god_map.world.groups[better_pose.default_env_name].movable_joint_names:
        joint = god_map.world.joints[joint_name]
        if isinstance(joint, OneDofJoint):
            js[str(joint.free_variable.name)] = 0.0
    better_pose.set_env_state(js)
    base_pose = PoseStamped()
    base_pose.header.frame_id = 'iai_apartment/side_B'
    base_pose.pose.position.x = 1.5
    base_pose.pose.position.y = 2.4
    base_pose.pose.orientation.w = 1
    base_pose = better_pose.transform_msg(god_map.world.root_link_name, base_pose)
    better_pose.teleport_base(base_pose)
    return better_pose

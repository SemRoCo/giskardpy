import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_about_axis

import giskardpy.utils.tfwrapper as tf
from giskardpy import identifier
from giskardpy.god_map import GodMap
from giskardpy.model.joints import OneDofJoint
from giskardpy.utils import logging
from giskardpy.utils.utils import launch_launchfile
from utils_for_tests import GiskardTestWrapper


@pytest.fixture(scope='module')
def ros(request):
    try:
        logging.loginfo('deleting tmp test folder')
        # shutil.rmtree(folder_name)
    except Exception:
        pass

        logging.loginfo('init ros')
    rospy.init_node('tests')
    tf.init(60)

    def kill_ros():
        GodMap().get_data(identifier.tree_manager).render()
        logging.loginfo('shutdown ros')
        rospy.signal_shutdown('die')
        try:
            logging.loginfo('deleting tmp test folder')
            # shutil.rmtree(folder_name)
        except Exception:
            pass

    try:
        rospy.get_param('kitchen_description')
    except:
        try:
            launch_launchfile('package://iai_kitchen/launch/upload_kitchen_obj.launch')
        except:
            logging.logwarn('iai_apartment not found')
    try:
        rospy.get_param('apartment_description')
    except:
        try:
            launch_launchfile('package://iai_apartment/launch/upload_apartment.launch')
        except:
            logging.logwarn('iai_kitchen not found')
    request.addfinalizer(kill_ros)


@pytest.fixture()
def resetted_giskard(giskard: GiskardTestWrapper) -> GiskardTestWrapper:
    logging.loginfo('resetting giskard')
    giskard.restart_ticking()
    if giskard.is_standalone() and giskard.has_odometry_joint():
        zero = PoseStamped()
        zero.header.frame_id = 'map'
        zero.pose.orientation.w = 1
        giskard.set_seed_odometry(zero)
    giskard.reset()
    return giskard


@pytest.fixture()
def zero_pose(resetted_giskard: GiskardTestWrapper) -> GiskardTestWrapper:
    if resetted_giskard.is_standalone():
        resetted_giskard.set_seed_configuration(resetted_giskard.default_pose)
        resetted_giskard.allow_all_collisions()
    else:
        resetted_giskard.allow_all_collisions()
        resetted_giskard.set_joint_goal(resetted_giskard.default_pose)
    resetted_giskard.plan_and_execute()
    return resetted_giskard


@pytest.fixture()
def better_pose(resetted_giskard: GiskardTestWrapper) -> GiskardTestWrapper:
    if resetted_giskard.is_standalone():
        resetted_giskard.set_seed_configuration(resetted_giskard.better_pose)
        resetted_giskard.allow_all_collisions()
    else:
        resetted_giskard.allow_all_collisions()
        resetted_giskard.set_joint_goal(resetted_giskard.better_pose)
    resetted_giskard.plan_and_execute()
    return resetted_giskard


@pytest.fixture()
def kitchen_setup(better_pose: GiskardTestWrapper) -> GiskardTestWrapper:
    better_pose.kitchen_name = 'iai_kitchen'
    if better_pose.is_standalone():
        kitchen_pose = PoseStamped()
        kitchen_pose.header.frame_id = str(better_pose.default_root)
        kitchen_pose.pose.orientation.w = 1
        better_pose.add_urdf(name=better_pose.kitchen_name,
                             urdf=rospy.get_param('kitchen_description'),
                             pose=kitchen_pose)
    else:
        kitchen_pose = tf.lookup_pose('map', 'iai_kitchen/world')
        better_pose.add_urdf(name=better_pose.kitchen_name,
                             urdf=rospy.get_param('kitchen_description'),
                             pose=kitchen_pose,
                             js_topic='/kitchen/joint_states',
                             set_js_topic='/kitchen/cram_joint_states')
    js = {}
    for joint_name in better_pose.world.groups[better_pose.kitchen_name].movable_joint_names:
        joint = better_pose.world.joints[joint_name]
        if isinstance(joint, OneDofJoint):
            if better_pose.is_standalone():
                js[str(joint.free_variable.name)] = 0.0
            else:
                js[str(joint.free_variable.name.short_name)] = 0.0
    better_pose.set_kitchen_js(js)
    return better_pose


@pytest.fixture()
def apartment_setup(better_pose: GiskardTestWrapper) -> GiskardTestWrapper:
    better_pose.environment_name = 'iai_apartment'
    if better_pose.is_standalone():
        kitchen_pose = PoseStamped()
        kitchen_pose.header.frame_id = str(better_pose.default_root)
        kitchen_pose.pose.orientation.w = 1
        better_pose.add_urdf(name=better_pose.environment_name,
                             urdf=rospy.get_param('apartment_description'),
                             pose=kitchen_pose)
    else:
        better_pose.add_urdf(name=better_pose.environment_name,
                             urdf=rospy.get_param('apartment_description'),
                             pose=tf.lookup_pose('map', 'iai_apartment/apartment_root'),
                             js_topic='/apartment_joint_states',
                             set_js_topic='/iai_kitchen/cram_joint_states')
    js = {}
    for joint_name in better_pose.world.groups[better_pose.environment_name].movable_joint_names:
        joint = better_pose.world.joints[joint_name]
        if isinstance(joint, OneDofJoint):
            js[str(joint.free_variable.name)] = 0.0
    better_pose.set_apartment_js(js)
    base_pose = PoseStamped()
    base_pose.header.frame_id = 'iai_apartment/side_B'
    base_pose.pose.position.x = 1.5
    base_pose.pose.position.y = 2.4
    base_pose.pose.orientation.w = 1
    base_pose = better_pose.transform_msg(better_pose.world.root_link_name, base_pose)
    better_pose.set_localization(base_pose)
    return better_pose

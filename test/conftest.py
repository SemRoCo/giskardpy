import pytest
import rospy

import giskardpy.utils.tfwrapper as tf
from giskardpy.utils import logging
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
        logging.loginfo('shutdown ros')
        rospy.signal_shutdown(u'die')
        try:
            logging.loginfo('deleting tmp test folder')
            # shutil.rmtree(folder_name)
        except Exception:
            pass

    request.addfinalizer(kill_ros)

@pytest.fixture()
def resetted_giskard(giskard):
    """
    :type giskard: PR2
    """
    logging.loginfo('resetting giskard')
    giskard.reset()
    return giskard


@pytest.fixture()
def zero_pose(resetted_giskard):
    """
    :type resetted_giskard: GiskardTestWrapper
    """
    resetted_giskard.allow_all_collisions()
    resetted_giskard.set_joint_goal(resetted_giskard.default_pose)
    resetted_giskard.plan_and_execute()
    return resetted_giskard


@pytest.fixture()
def better_pose(resetted_giskard):
    """
    :type resetted_giskard: GiskardTestWrapper
    :rtype: GiskardTestWrapper
    """
    resetted_giskard.set_joint_goal(resetted_giskard.better_pose)
    resetted_giskard.allow_all_collisions()
    resetted_giskard.plan_and_execute()
    return resetted_giskard


@pytest.fixture()
def kitchen_setup(better_pose):
    """
    :type better_pose: GiskardTestWrapper
    :return: GiskardTestWrapper
    """
    object_name = u'kitchen'
    better_pose.add_urdf(object_name, rospy.get_param('kitchen_description'),
                              tf.lookup_pose(u'map', 'iai_kitchen/world'), '/kitchen/joint_states',
                              set_js_topic='/kitchen/cram_joint_states')
    js = {str(k): 0.0 for k in better_pose.world.groups[object_name].movable_joints}
    better_pose.set_kitchen_js(js)
    return better_pose

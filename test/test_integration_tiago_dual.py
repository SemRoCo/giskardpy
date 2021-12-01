import pytest
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_about_axis

from utils_for_tests import TiagoDual


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = TiagoDual()
    request.addfinalizer(c.tear_down)
    return c


class TestCartGoals(object):
    def test_drive(self, zero_pose):
        """
        :type zero_pose: TiagoDual
        """
        zero_pose.allow_all_collisions()
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        # goal.pose.position.x = 1
        goal.pose.position.y = 1
        goal.pose.orientation.w = 1
        # goal.pose.orientation = Quaternion(*quaternion_about_axis(1, [0, 0, 1]))
        # zero_pose.move_base(goal)
        zero_pose.set_translation_goal(goal, 'base_footprint', 'odom')
        zero_pose.plan_and_execute()

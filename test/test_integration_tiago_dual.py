import pytest
from geometry_msgs.msg import PoseStamped

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
        zero_pose.allow_self_collision()
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 1
        goal.pose.orientation.w = 1
        zero_pose.move_base(goal)
        zero_pose.plan_and_execute()

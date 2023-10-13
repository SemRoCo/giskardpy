import traceback
from py_trees import Status

from giskardpy.exceptions import InvalidGoalException
from giskardpy.goals.base_traj_follower import BaseTrajFollower
from giskardpy.goals.monitors.monitors import TimeAbove
from giskardpy.god_map import god_map
from giskardpy.model.joints import OmniDrive, DiffDrive
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.logging import loginfo
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class ParseActionGoal(GiskardBehavior):
    @record_time
    @profile
    def __init__(self, name):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        loginfo('Parsing goal message.')
        move_goal = god_map.goal_msg
        god_map.goal_id += 1
        try:
            god_map.monitor_manager.parse_monitors(move_goal.monitors)
            god_map.motion_goal_manager.parse_motion_goals(move_goal.goals)
        except AttributeError:
            traceback.print_exc()
            raise InvalidGoalException('Couldn\'t transform goal')
        except Exception as e:
            raise e
        if god_map.is_collision_checking_enabled():
            god_map.motion_goal_manager.parse_collision_entries(move_goal.collisions)
        loginfo('Done parsing goal message.')
        return Status.SUCCESS


class AddBaseTrajFollowerGoal(GiskardBehavior):
    def __init__(self, name: str = 'add base traj goal'):
        super().__init__(name)
        joints = god_map.world.search_for_joint_of_type((OmniDrive, DiffDrive))
        assert len(joints) == 1
        self.joint = joints[0]

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        goal = BaseTrajFollower(self.joint.name)
        time_monitor = TimeAbove(threshold=god_map.trajectory.length_in_seconds)
        goal.connect_to_end(time_monitor)
        god_map.monitor_manager.add_monitor(time_monitor)
        god_map.motion_goal_manager.add_motion_goal(goal)
        return Status.SUCCESS


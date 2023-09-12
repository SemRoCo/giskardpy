from typing import Dict

from giskardpy.python_interface.low_level_python_interface import LowLevelGiskardWrapper


class GiskardWrapper(LowLevelGiskardWrapper):
    def send_and_reach_joint_goal(self, goal_state: Dict[str, float]):
        self.add_monitor()
        self.set_joint_goal(goal_state)

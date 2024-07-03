#!/usr/bin/env python
import rospy

from giskardpy.configs.behavior_tree_config import ClosedLoopBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.pr2 import PR2CollisionAvoidance, WorldWithPR2Config, PR2VelocityIAIInterface
from giskardpy.configs.qp_controller_config import QPControllerConfig, SupportedQPSolver


class WorldWithPR2ConfigBlue(WorldWithPR2Config):

    def setup(self):
        super().setup()
        self.set_default_color(20 / 255, 27.1 / 255, 80 / 255, 0.2)


if __name__ == '__main__':
    rospy.init_node('giskard')
    giskard = Giskard(world_config=WorldWithPR2ConfigBlue(),
                      collision_avoidance_config=PR2CollisionAvoidance(),
                      robot_interface_config=PR2VelocityIAIInterface(),
                      behavior_tree_config=ClosedLoopBTConfig(debug_mode=True),
                      qp_controller_config=QPControllerConfig())
    giskard.live()

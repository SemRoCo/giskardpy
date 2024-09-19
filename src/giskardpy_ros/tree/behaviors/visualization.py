from typing import Optional

import py_trees
import rospy
from line_profiler import profile

from giskardpy.god_map import god_map
from giskardpy_ros.ros1.ros_msg_visualization import ROSMsgVisualization, VisualizationMode
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.blackboard_utils import catch_and_raise_to_blackboard


class VisualizationBehavior(GiskardBehavior):
    @profile
    def __init__(self,
                 mode: VisualizationMode,
                 name: str = 'visualization marker',
                 ensure_publish: bool = False):
        super().__init__(name)
        self.ensure_publish = ensure_publish
        self.visualizer = ROSMsgVisualization(mode=mode)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        self.visualizer.publish_markers()
        if self.ensure_publish:
            rospy.sleep(0.1)
        # rospy.sleep(0.01)
        return py_trees.common.Status.SUCCESS


class VisualizeTrajectory(GiskardBehavior):
    @profile
    def __init__(self,
                 mode: VisualizationMode = VisualizationMode.CollisionsDecomposed,
                 name: Optional[str] = None,
                 ensure_publish: bool = False):
        super().__init__(name)
        self.ensure_publish = ensure_publish
        self.visualizer = god_map.ros_visualizer
        self.every_x = 10

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        self.visualizer.publish_trajectory_markers(trajectory=god_map.trajectory,
                                                   every_x=self.every_x)
        return py_trees.common.Status.SUCCESS

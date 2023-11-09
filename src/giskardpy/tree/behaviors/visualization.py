import py_trees
import rospy
from visualization_msgs.msg import Marker, MarkerArray

from giskardpy.model.ros_msg_visualization import ROSMsgVisualization
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class VisualizationBehavior(GiskardBehavior):
    @profile
    def __init__(self,
                 name: str = 'visualization marker',
                 ensure_publish: bool = False,
                 use_decomposed_meshes: bool = True):
        super().__init__(name)
        self.ensure_publish = ensure_publish
        self.visualizer = ROSMsgVisualization(use_decomposed_meshes=use_decomposed_meshes)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        self.visualizer.publish_markers()
        if self.ensure_publish:
            rospy.sleep(0.1)
        rospy.sleep(0.01)
        return py_trees.common.Status.SUCCESS

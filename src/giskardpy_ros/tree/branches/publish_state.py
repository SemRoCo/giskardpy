from typing import Optional

from py_trees import Sequence

from giskardpy_ros.tree.behaviors.debug_marker_publisher import DebugMarkerPublisher
from giskardpy_ros.tree.behaviors.publish_debug_expressions import PublishDebugExpressions
from giskardpy_ros.tree.behaviors.publish_feedback import PublishFeedback
from giskardpy_ros.tree.behaviors.publish_joint_states import PublishJointState
from giskardpy_ros.tree.behaviors.tf_publisher import TfPublishingModes, TFPublisher
from giskardpy_ros.tree.behaviors.visualization import VisualizationBehavior
from giskardpy.utils.decorators import toggle_on, toggle_off


class PublishState(Sequence):
    visualization_behavior: Optional[VisualizationBehavior]

    def __init__(self, name: str = 'publish state'):
        super().__init__(name)
        self.visualization_behavior = None

    @toggle_on('visualization_marker_behavior')
    def add_visualization_marker_behavior(self, use_decomposed_meshes: bool = True):
        if self.visualization_behavior is None:
            self.visualization_behavior = VisualizationBehavior(use_decomposed_meshes=use_decomposed_meshes)
        self.add_child(self.visualization_behavior)

    @toggle_off('visualization_marker_behavior')
    def remove_visualization_marker_behavior(self):
        self.remove_child(self.visualization_behavior)

    def add_debug_marker_publisher(self):
        self.add_child(DebugMarkerPublisher())

    def add_publish_feedback(self):
        self.add_child(PublishFeedback())

    def add_tf_publisher(self, include_prefix: bool = False, tf_topic: str = 'tf',
                         mode: TfPublishingModes = TfPublishingModes.attached_and_world_objects):
        node = TFPublisher('publish tf', mode=mode, tf_topic=tf_topic, include_prefix=include_prefix)
        self.add_child(node)

    def add_qp_data_publisher(self, publish_lb: bool = False, publish_ub: bool = False, publish_lbA: bool = False,
                              publish_ubA: bool = False, publish_bE: bool = False, publish_Ax: bool = False,
                              publish_Ex: bool = False, publish_xdot: bool = False, publish_weights: bool = False,
                              publish_g: bool = False, publish_debug: bool = False):
        node = PublishDebugExpressions(publish_lb=publish_lb,
                                       publish_ub=publish_ub,
                                       publish_xdot=publish_xdot,
                                       publish_lbA=publish_lbA,
                                       publish_ubA=publish_ubA,
                                       publish_Ax=publish_Ax,
                                       publish_Ex=publish_Ex,
                                       publish_bE=publish_bE,
                                       publish_weights=publish_weights,
                                       publish_g=publish_g,
                                       publish_debug=publish_debug)
        self.add_child(node)

    def add_joint_state_publisher(self, topic_name: Optional[str] = None, include_prefix: bool = False,
                                  only_prismatic_and_revolute: bool = True):
        node = PublishJointState(include_prefix=include_prefix, topic_name=topic_name,
                                 only_prismatic_and_revolute=only_prismatic_and_revolute)
        self.add_child(node)

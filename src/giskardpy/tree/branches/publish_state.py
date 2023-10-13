from py_trees import Sequence

from giskardpy.tree.behaviors.debug_marker_publisher import DebugMarkerPublisher
from giskardpy.tree.behaviors.publish_debug_expressions import PublishDebugExpressions
from giskardpy.tree.behaviors.tf_publisher import TfPublishingModes, TFPublisher
from giskardpy.tree.behaviors.visualization import VisualizationBehavior


class PublishState(Sequence):
    def __init__(self, name: str = 'publish state'):
        super().__init__(name)

    def add_visualization_marker_behavior(self, use_decomposed_meshes: bool = True):
        self.add_child(VisualizationBehavior(name='visualization', use_decomposed_meshes=use_decomposed_meshes))

    def add_debug_marker_publisher(self):
        self.add_child(DebugMarkerPublisher())

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

from py_trees import Sequence

from giskardpy.tree.behaviors.tf_publisher import TfPublishingModes, TFPublisher
from giskardpy.tree.behaviors.visualization import VisualizationBehavior


class PublishState(Sequence):
    def __init__(self, name: str = 'publish state'):
        super().__init__(name)

    def add_visualization_marker_behavior(self, use_decomposed_meshes: bool = True):
        self.add_child(VisualizationBehavior(name='visualization', use_decomposed_meshes=use_decomposed_meshes))

    def add_tf_publisher(self, include_prefix: bool = False, tf_topic: str = 'tf',
                         mode: TfPublishingModes = TfPublishingModes.attached_and_world_objects):
        node = TFPublisher('publish tf', mode=mode, tf_topic=tf_topic, include_prefix=include_prefix)
        self.add_child(node)

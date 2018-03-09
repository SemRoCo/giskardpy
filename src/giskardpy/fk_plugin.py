from geometry_msgs.msg import PoseStamped

from giskardpy.plugin import IOPlugin
from giskardpy.tfwrapper import TfWrapper


class FKPlugin(IOPlugin):
    def __init__(self, root, tip):
        self.root = root
        self.tip = tip
        super(FKPlugin, self).__init__()

    def get_readings(self):
        p = PoseStamped()
        p.header.frame_id = self.tip
        p.pose.orientation.w = 1
        return {'fk_{}'.format(self.tip): self.tf.transform_pose(self.root, p)}

    def update(self):
        super(FKPlugin, self).update()

    def start(self, databus):
        self.tf = TfWrapper()
        super(FKPlugin, self).start(databus)

    def stop(self):
        pass

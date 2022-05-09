from giskardpy.tree.behaviors.send_trajectory import SendFollowJointTrajectory


class FollowJointTrajectoryInterface:
    def __init__(self,
                 namespace: str,
                 state_topic: str,
                 fill_velocity_values: bool = False):
        self.namespace = namespace
        self.state_topic = state_topic
        self.fill_velocity_values = fill_velocity_values

    def make_plugin(self):
        return SendFollowJointTrajectory(self.namespace, self.namespace, self.state_topic,
                                         fill_velocity_values=self.fill_velocity_values)

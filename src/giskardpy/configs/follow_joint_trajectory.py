from giskardpy.tree.behaviors.send_trajectory import SendFollowJointTrajectory


class FollowJointTrajectoryInterface:
    def __init__(self,
                 action_namespace: str,
                 group_name: str,
                 state_topic: str,
                 fill_velocity_values: bool = False):
        self.group_name = group_name
        self.namespace = action_namespace
        self.state_topic = state_topic
        self.fill_velocity_values = fill_velocity_values

    def make_plugin(self):
        return SendFollowJointTrajectory(group_name=self.group_name,
                                         action_namespace=self.namespace,
                                         state_topic=self.state_topic,
                                         fill_velocity_values=self.fill_velocity_values)

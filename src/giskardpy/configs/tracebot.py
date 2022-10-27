from giskardpy.configs.default_giskard import Giskard


class TracebotMujoco(Giskard):
    def __init__(self):
        super().__init__()
        self.robot_interface_configs.joint_state_topic = 'tracebot/joint_states'
        self.add_follow_joint_trajectory_server(namespace='/tracebot/whole_body_controller/follow_joint_trajectory',
                                                state_topic='/tracebot/whole_body_controller/state')

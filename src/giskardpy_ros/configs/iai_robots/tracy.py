import numpy as np

from giskardpy.model.collision_avoidance_config import LoadSelfCollisionMatrixConfig
from giskardpy.model.world_config import WorldWithFixedRobot
from giskardpy_ros.configs.robot_interface_config import RobotInterfaceConfig, StandAloneRobotInterfaceConfig
from giskardpy.model.collision_world_syncer import CollisionCheckerLib
from giskardpy.data_types.data_types import Derivatives


class TracyWorldConfig(WorldWithFixedRobot):
    def __init__(self):
        super().__init__({Derivatives.velocity: 0.2,
                          Derivatives.acceleration: np.inf,
                          Derivatives.jerk: 15})


class TracyCollisionAvoidanceConfig(LoadSelfCollisionMatrixConfig):
    def __init__(self, collision_checker: CollisionCheckerLib = CollisionCheckerLib.bpb):
        super().__init__('package://giskardpy/self_collision_matrices/iai/tracy.srdf',
                         collision_checker)


class TracyJointTrajServerMujocoInterface(RobotInterfaceConfig):
    def setup(self):
        self.sync_joint_state_topic('joint_states')
        self.add_follow_joint_trajectory_server(
            namespace='/left_arm/scaled_pos_joint_traj_controller_left')
        self.add_follow_joint_trajectory_server(
            namespace='/right_arm/scaled_pos_joint_traj_controller_right')


class TracyStandAloneRobotInterfaceConfig(StandAloneRobotInterfaceConfig):
    def __init__(self):
        super().__init__([
            'left_shoulder_pan_joint',
            'left_shoulder_lift_joint',
            'left_elbow_joint',
            'left_wrist_1_joint',
            'left_wrist_2_joint',
            'left_wrist_3_joint',
            'right_shoulder_pan_joint',
            'right_shoulder_lift_joint',
            'right_elbow_joint',
            'right_wrist_1_joint',
            'right_wrist_2_joint',
            'right_wrist_3_joint',
        ])

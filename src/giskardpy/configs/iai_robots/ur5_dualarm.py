import numpy as np

from giskardpy.configs.collision_avoidance_config import CollisionAvoidanceConfig
from giskardpy.configs.robot_interface_config import RobotInterfaceConfig, StandAloneRobotInterfaceConfig
from giskardpy.configs.world_config import WorldWithFixedRobot
from giskardpy.model.collision_world_syncer import CollisionCheckerLib
from giskardpy.data_types import Derivatives


class UR5DualArmJointTrajServerInterface(RobotInterfaceConfig):
    def setup(self):
        self.sync_joint_state_topic('joint_states')
        self.add_follow_joint_trajectory_server(
            namespace='/left_arm/pos_joint_traj_controller_left')
        self.add_follow_joint_trajectory_server(
            namespace='/right_arm/pos_joint_traj_controller_right')


class UR5DualArmVelocityInterface(RobotInterfaceConfig):
    def setup(self):
        self.sync_joint_state_topic('joint_states')
        self.add_joint_velocity_group_controller(namespace='left_arm/joint_group_vel_controller_left')
        self.add_joint_velocity_group_controller(namespace='right_arm/joint_group_vel_controller_right')


class UR5DualArmCollisionAvoidanceConfig(CollisionAvoidanceConfig):
    def __init__(self):
        super().__init__()

    def setup(self):
        self.load_self_collision_matrix('package://giskardpy/self_collision_matrices/iai/ur5dualarm.srdf')


class UR5DualArmWorldConfig(WorldWithFixedRobot):
    def __init__(self):
        super().__init__({Derivatives.velocity: 0.2,
                          Derivatives.acceleration: np.inf,
                          Derivatives.jerk: 15})

    def setup(self):
        super().setup()
        self.set_joint_limits(limit_map={Derivatives.velocity: 3,
                                         Derivatives.jerk: 80},
                              joint_name='left_wrist_3_joint')

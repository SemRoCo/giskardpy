from collections import defaultdict

from giskardpy.configs.default_config import GiskardConfig, CollisionAvoidanceConfig
from giskardpy.configs.drives import OmniDriveCmdVelInterface
from giskardpy.configs.follow_joint_trajectory import FollowJointTrajectoryInterface


class SuperCloseCollisionAvoidance(CollisionAvoidanceConfig):
    def __init__(self):
        super().__init__(soft_threshold=0.00001,
                         hard_threshold=0.0)


class CloseCollisionAvoidance(CollisionAvoidanceConfig):
    def __init__(self):
        super().__init__(soft_threshold=0.02,
                         hard_threshold=0.005)


class Donbot(GiskardConfig):
    ignored_self_collisions = [
        ['ur5_forearm_link', 'ur5_wrist_3_link'],
        ['ur5_base_link', 'ur5_upper_arm_link']
    ]
    add_self_collisions = [
        ['plate', 'ur5_upper_arm_link']
    ]
    external_collision_avoidance = defaultdict(CollisionAvoidanceConfig.init_100mm,
                                               {
                                                   'brumbrum': CollisionAvoidanceConfig(
                                                       number_of_repeller=2,
                                                       soft_threshold=0.1,
                                                       hard_threshold=0.05,
                                                   ),
                                               })
    self_collision_avoidance = defaultdict(CollisionAvoidanceConfig,
                                           {
                                               'ur5_wrist_1_link': CloseCollisionAvoidance(),
                                               'ur5_wrist_2_link': CloseCollisionAvoidance(),
                                               'ur5_wrist_3_link': CloseCollisionAvoidance(),
                                               'ur5_forearm_link': CloseCollisionAvoidance(),
                                               'ur5_upper_arm_link': CloseCollisionAvoidance(),
                                               'gripper_gripper_left_link': SuperCloseCollisionAvoidance(),
                                               'gripper_gripper_right_link': SuperCloseCollisionAvoidance(),
                                           })

    def __init__(self):
        super().__init__()
        self.plugin_config['SyncTfFrames'] = {
            'frames': [['map', 'odom']]
        }
        self.plugin_config['SyncOdometry'] = {
            'odometry_topic': '/donbot/base_footprint'
        }
        self.follow_joint_trajectory_interfaces = [
            FollowJointTrajectoryInterface(namespace='/donbot/whole_body_controller/follow_joint_trajectory',
                                           state_topic='/donbot/whole_body_controller/state')
        ]
        self.drive_interface = OmniDriveCmdVelInterface(
            cmd_vel_topic='/donbot/cmd_vel',
            parent_link_name='odom',
            child_link_name='base_footprint')

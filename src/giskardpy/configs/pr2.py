from collections import defaultdict

from giskardpy.configs.default_config import GiskardConfig, CollisionAvoidanceConfig


class EndEffector4(CollisionAvoidanceConfig):
    def __init__(self):
        super().__init__(number_of_repeller=4,
                         soft_threshold=0.05,
                         hard_threshold=0.0,
                         max_velocity=0.2)

class EndEffector2(CollisionAvoidanceConfig):
    def __init__(self):
        super().__init__(number_of_repeller=2,
                         soft_threshold=0.05,
                         hard_threshold=0.0,
                         max_velocity=0.2)

class PR2(GiskardConfig):
    ignored_self_collisions = [
        'bl_caster_l_wheel_link',
        'bl_caster_r_wheel_link',
        'bl_caster_rotation_link',
        'br_caster_l_wheel_link',
        'br_caster_r_wheel_link',
        'br_caster_rotation_link',
        'fl_caster_l_wheel_link',
        'fl_caster_r_wheel_link',
        'fl_caster_rotation_link',
        'fr_caster_l_wheel_link',
        'fr_caster_r_wheel_link',
        'fr_caster_rotation_link',
        'l_shoulder_lift_link',
        'r_shoulder_lift_link',
        'base_link',
    ]
    external_collision_avoidance = defaultdict(CollisionAvoidanceConfig.init_100mm,
                                               {
                                                   'r_wrist_roll_joint': EndEffector4(),
                                                   'l_wrist_roll_joint': EndEffector4(),
                                                   'r_elbow_flex_joint': CollisionAvoidanceConfig.init_50mm(),
                                                   'l_elbow_flex_joint': CollisionAvoidanceConfig.init_50mm(),
                                                   'r_wrist_flex_joint': EndEffector2(),
                                                   'l_wrist_flex_joint': EndEffector2(),
                                                   'r_forearm_roll_joint': CollisionAvoidanceConfig.init_25mm(),
                                                   'l_forearm_roll_joint': CollisionAvoidanceConfig.init_25mm()
                                               })

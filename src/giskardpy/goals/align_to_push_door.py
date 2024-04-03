from typing import Optional
import numpy as np

from geometry_msgs.msg import Vector3Stamped, PointStamped, Quaternion
from std_msgs.msg import ColorRGBA

from giskardpy import casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.monitors.monitors import ExpressionMonitor
from giskardpy.tasks.task import WEIGHT_BELOW_CA
from giskardpy.utils import tfwrapper as tf


class AlignToPushDoor(Goal):
    # start condition - door is open a little bit
    # hold condition - door is open a little bit
    # end condition - gripper is in the gap between the open door and the dishwasher
    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 door_object: str,
                 # you will need the intermediate point in the direction the object opens. In dishwasher along z axis
                 # in door along y axis?
                 door_height: float,  #
                 object_joint_name: str,
                 tip_gripper_axis: Vector3Stamped,
                 object_rotation_axis: Vector3Stamped,
                 root_group: str,
                 tip_group: str,
                 reference_linear_velocity: float = 0.1,
                 reference_angular_velocity: float = 0.5,
                 weight: float = WEIGHT_BELOW_CA,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
        """
        The objective is to reach an intermediate point before pushing the door
        """
        self.root = god_map.search_for_link_name(root_link, root_group)
        self.tip = god_map.search_for_link_name(tip_link, tip_group)
        self.door_object = god_map.search_for_link_name(door_object)
        self.object_joint_angle = \
            god_map.state.to_position_dict()[god_map.search_for_joint_name(object_joint_name)]

        tip_gripper_axis.header.frame_id = self.tip
        tip_gripper_axis.vector = tf.normalize(tip_gripper_axis.vector)
        object_rotation_axis.header.frame_id = self.door_object
        object_rotation_axis.vector = tf.normalize(object_rotation_axis.vector)

        self.tip_gripper_axis = tip_gripper_axis
        self.object_rotation_axis = object_rotation_axis
        self.root_P_door_object = PointStamped()
        self.root_P_door_object.header.frame_id = self.root

        # Pose of the articulated object is at the center of the axis along which it rotates and not at the center of
        # the object.
        # self.root_T_door = godmap.compute_fk_pose(self.root, self.door_object)
        # self.root_T_door = door_pose_before_rotation
        # self.root_T_door.header.frame_id = self.root
        # self.root_P_door_object.point = self.root_T_door.pose.position
        self.door_height = door_height
        # self.door_length = door_length

        # self.object_rotation_angle = object_rotation_angle
        self.reference_linear_velocity = reference_linear_velocity
        self.reference_angular_velocity = reference_angular_velocity
        self.weight = weight
        if name is None:
            name = f'{self.__class__.__name__}/{self.root}/{self.tip}'
        super().__init__(name=name)

        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        root_T_object = god_map.world.compose_fk_expression(self.root, self.door_object)
        object_V_object_rotation_axis = cas.Vector3(self.object_rotation_axis)
        tip_V_tip_grasp_axis = cas.Vector3(self.tip_gripper_axis)
        root_V_object_rotation_axis = cas.dot(root_T_object, object_V_object_rotation_axis)
        root_V_tip_grasp_axis = cas.dot(root_T_tip, tip_V_tip_grasp_axis)

        root_P_object = god_map.world.compute_fk_pose(self.root, self.door_object)

        root_P_intermediate_point = PointStamped()
        root_P_intermediate_point.header.frame_id = self.root
        root_P_intermediate_point.point.x = root_P_object.pose.position.x
        root_P_intermediate_point.point.y = root_P_object.pose.position.y
        root_P_intermediate_point.point.z = root_P_object.pose.position.z + self.door_height

        door_T_root = god_map.world.compute_fk_pose(self.door_object, self.root)

        # point w.r.t door
        door_P_intermediate_point = cas.dot(cas.TransMatrix(door_T_root), cas.Point3(root_P_intermediate_point))
        desired_angle = self.object_joint_angle * 0.5  # just chose 1/2 of the goal angle

        # find rotated point in local frame

        q = Quaternion()
        rot_vector = np.array([self.object_rotation_axis.vector.x, self.object_rotation_axis.vector.y,
                               self.object_rotation_axis.vector.z])
        q.x = rot_vector[0] * np.sin(desired_angle / 2),
        q.y = rot_vector[1] * np.sin(desired_angle / 2),
        q.z = rot_vector[2] * np.sin(desired_angle / 2),
        q.w = np.cos(desired_angle / 2)
        rot_mat = cas.RotationMatrix(q)

        door_P_rotated_point = cas.dot(rot_mat, cas.Point3(door_P_intermediate_point))

        root_P_rotated_point = cas.dot(cas.TransMatrix(root_T_object), cas.Point3(door_P_rotated_point))

        god_map.debug_expression_manager.add_debug_expression('goal_point', cas.Point3(root_P_rotated_point),
                                                              color=ColorRGBA(0, 0.5, 0.5, 1))

        god_map.debug_expression_manager.add_debug_expression('root_V_grasp_axis', root_V_tip_grasp_axis)
        god_map.debug_expression_manager.add_debug_expression('root_V_object_axis', root_V_object_rotation_axis)

        door_open = cas.greater_equal(self.object_joint_angle, 0.349)
        door_opened_monitor = ExpressionMonitor(name='door opened', stay_true=True, start_condition=start_condition)
        self.add_monitor(door_opened_monitor)
        door_opened_monitor.expression = door_open

        align_to_push_task = self.create_and_add_task('align_to_push_door')
        align_to_push_task.add_point_goal_constraints(frame_P_current=root_T_tip.to_position(),
                                                      frame_P_goal=cas.Point3(root_P_rotated_point),
                                                      reference_velocity=self.reference_linear_velocity,
                                                      weight=self.weight)

        align_to_push_task.add_vector_goal_constraints(frame_V_current=cas.Vector3(root_V_tip_grasp_axis),
                                                       frame_V_goal=cas.Vector3(root_V_object_rotation_axis),
                                                       reference_velocity=self.reference_angular_velocity,
                                                       weight=self.weight)

        align_to_push_task.start_condition = door_opened_monitor
        # align_to_push_task.end_condition = tip_aligned_monitor
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)

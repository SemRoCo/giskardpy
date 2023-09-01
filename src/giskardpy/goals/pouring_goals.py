import math

import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w
from giskardpy.goals.goal import Goal, WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE
from typing import Optional
from geometry_msgs.msg import Vector3Stamped, PointStamped, QuaternionStamped
from giskardpy.goals.cartesian_goals import CartesianOrientation, RotationVelocityLimit
from giskardpy.goals.align_planes import AlignPlanes


# Todo: the HoldObjectUpright constraint could be realized using the align planes goal
# Info: a pouring action can be realized in three steps
#       1. KeepObjectUpright & KeepObjectAbovePlane
#       2. TiltObject & KeepObjectAbovePlane
#       3. KeepObjectUpright & KeepObjectAbovePlane


class KeepObjectAbovePlane(Goal):
    # Todo: works if the height axis lies on a main axis. For example, y=z=0.5 and the gripper moves to teh right
    #       position, but the body moves weird and trajectory is too long and distance and distance2 converge to the
    #       same value => the constraints are not decoupled correctly

    def __init__(self,
                 object_link: str,
                 plane_center_point: PointStamped,
                 lower_distance: float,
                 root_link: str,
                 height_axis: Vector3Stamped = None,
                 upper_distance: Optional[float] = None,
                 plane_radius: float = 0.0,
                 max_velocity: float = 0.3,
                 weight: float = WEIGHT_BELOW_CA,
                 object_group: Optional[str] = None,
                 root_group: Optional[str] = None,
                 ):
        """
        Keeps the object link between a lower and upper distance from the plane
        center point measured along the height axis. Given a plane radius,
        the object link is placed above a circular plane instead of above a point.
        """
        super().__init__()
        self.weight = weight
        self.max_velocity = max_velocity
        self.root = self.world.search_for_link_name(root_link, root_group)
        self.tip = self.world.search_for_link_name(object_link, object_group)
        self.lower_distance = lower_distance
        if upper_distance is None:
            self.upper_distance = lower_distance
        else:
            self.upper_distance = upper_distance
        self.plane_radius = plane_radius
        self.root_P_plane_center_point = self.transform_msg(self.root, plane_center_point)
        p = PointStamped()
        if height_axis is None:
            self.root_V_height_axis = w.Vector3([0, 0, 1])
            p.header.frame_id = root_link
        else:
            self.root_V_height_axis = self.transform_msg(self.root, height_axis)
            p.header.frame_id = height_axis.header.frame_id
        p.point.x = 0
        p.point.y = 0
        p.point.z = 0
        self.root_P_height_origin = self.transform_msg(self.root, p)

    def make_constraints(self):
        root_T_object = self.get_fk(self.root, self.tip)
        root_P_object = root_T_object.to_position()
        root_P_plane = w.Point3(self.root_P_plane_center_point)
        root_V_height_axis = w.Vector3(self.root_V_height_axis)
        root_P_height_origin = w.Point3(self.root_P_height_origin)

        def project_point_on_vector(point, vector):
            # returns a point
            return w.dot(point, vector) * vector

        root_P_object_proj = project_point_on_vector(root_P_object, root_V_height_axis)
        root_P_plane_proj = project_point_on_vector(root_P_plane, root_V_height_axis)
        distance = w.euclidean_distance(root_P_object_proj, root_P_plane_proj)

        # constrain object to be above plane
        self.add_inequality_constraint_vector(reference_velocities=[self.max_velocity] * 3,
                                              lower_errors=root_P_plane_proj[:3] - root_P_object_proj[:3],
                                              upper_errors=root_V_height_axis[:3],
                                              weights=[WEIGHT_ABOVE_CA * self.weight / WEIGHT_BELOW_CA] * 3,
                                              # WEIGHT needs to be that high, to work for start positions  where the tip is below the plane
                                              task_expression=root_P_object_proj[:3],
                                              names=['abovex', 'abovey', 'abovez'])
        # constrain object to be at a certain distance
        self.add_inequality_constraint(reference_velocity=self.max_velocity,
                                       lower_error=self.lower_distance - distance,
                                       upper_error=self.upper_distance - distance,
                                       task_expression=distance,
                                       weight=self.weight,
                                       name='distance')

        # constrain the x-y-plane to be above the point
        def project_point_onto_plane(point, plane_normal, plane_origin):
            v = point - plane_origin
            return point - project_point_on_vector(v, plane_normal)

        root_P_object_proj_plane = project_point_onto_plane(root_P_object, root_V_height_axis, root_P_height_origin)
        root_P_plane_proj_plane = project_point_onto_plane(root_P_plane, root_V_height_axis, root_P_height_origin)
        distance2 = w.euclidean_distance(root_P_plane_proj_plane, root_P_object_proj_plane)
        self.add_inequality_constraint(reference_velocity=self.max_velocity,
                                       lower_error=0 - distance2,
                                       upper_error=self.plane_radius - distance2,
                                       task_expression=distance2,
                                       weight=self.weight,
                                       name='distance2')
        # self.add_debug_expr('distancexy', distance2)
        # self.add_debug_expr('distance_height', distance)

        self.add_debug_expr('cupPosition', root_P_object)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root}/{self.tip}'


class TiltObject(Goal):
    # Todo: look into constraining the velocity to a specific value. What about ramp up and ramp down?
    def __init__(self,
                 object_link: str,
                 reference_link: str,
                 rotation_axis: Vector3Stamped,
                 rotation_velocity: float,
                 lower_angle: float,
                 root_link: str,
                 upper_angle: Optional[float] = None,
                 max_velocity: float = 0.3,
                 weight: float = WEIGHT_BELOW_CA,
                 object_group: Optional[str] = None,
                 root_group: Optional[str] = None,
                 name_extra=None
                 ):
        """
        Tilts the object link between lower and upper angles in degree around the
        rotation axis compared to the frame of the reference link
        """
        super().__init__()
        self.weight = weight
        self.max_velocity = max_velocity
        self.root = root_link
        self.tip = object_link
        self.lower_angle = lower_angle
        if upper_angle is None:
            self.upper_angle = lower_angle
        self.reference_link = reference_link
        self.ref_V_rotation_axis = self.transform_msg(self.reference_link, rotation_axis)
        self.rotation_velocity = rotation_velocity
        self.root_group = root_group
        self.object_group = object_group
        self.name_extra = name_extra

    def make_constraints(self):
        def get_quaternion_from_rotation_around_axis(angle, axis, axis_frame):
            q = QuaternionStamped()
            q.quaternion.w = math.cos(angle / 2)
            q.quaternion.x = math.sin(angle / 2) * axis.vector.x
            q.quaternion.y = math.sin(angle / 2) * axis.vector.y
            q.quaternion.z = math.sin(angle / 2) * axis.vector.z
            q.header.frame_id = axis_frame
            return q

        goal_orientation = get_quaternion_from_rotation_around_axis(self.lower_angle,
                                                                    self.ref_V_rotation_axis,
                                                                    self.reference_link)

        self.add_constraints_of_goal(CartesianOrientation(root_link=self.root,
                                                          root_group=self.root_group,
                                                          tip_link=self.tip,
                                                          tip_group=self.object_group,
                                                          goal_orientation=goal_orientation,
                                                          max_velocity=self.rotation_velocity,
                                                          reference_velocity=self.max_velocity,
                                                          weight=self.weight,
                                                          name_extra=self.name_extra))
        self.add_constraints_of_goal(RotationVelocityLimit(root_link=self.root,
                                                           root_group=self.root_group,
                                                           tip_link=self.tip,
                                                           tip_group=self.object_group,
                                                           max_velocity=self.rotation_velocity,
                                                           weight=self.weight,
                                                           hard=True,
                                                           name_extra=self.name_extra))
        root_V_y = w.Vector3([0, 1, 0])
        root_T_tip = self.get_fk(self.root, self.world.search_for_link_name(self.tip, self.object_group))
        root_V_tip_y = root_T_tip.to_rotation()[:, 1]
        angle = w.angle_between_vector(root_V_y, root_V_tip_y)
        self.add_debug_expr('angle', angle)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root}/{self.tip}/{self.name_extra}'


class KeepObjectUpright(Goal):
    def __init__(self,
                 object_link_axis: Vector3Stamped,
                 reference_link_axis: Vector3Stamped,
                 root_link: str,
                 allowed_error: Optional[float] = 0,
                 max_velocity: float = 0.3,
                 weight: float = WEIGHT_BELOW_CA):
        """
        Aligns the axis of the object frame and the reference frame.
        Maybe an allowed error for deviation in degrees is introduced.
        """
        super().__init__()
        self.weight = weight
        self.max_velocity = max_velocity
        self.root = root_link
        self.tip = object_link_axis.header.frame_id
        self.reference_link_axis = reference_link_axis
        self.allowed_error = allowed_error
        self.object_link_axis = object_link_axis

    def make_constraints(self):
        self.add_constraints_of_goal(AlignPlanes(root_link=self.root,
                                                 tip_link=self.tip,
                                                 goal_normal=self.reference_link_axis,
                                                 tip_normal=self.object_link_axis))

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.root}/{self.tip}'

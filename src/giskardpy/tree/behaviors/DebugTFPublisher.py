import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped, Quaternion, Point, Transform
from py_trees import Status
from tf.transformations import quaternion_from_matrix
from tf2_msgs.msg import TFMessage

import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.tfwrapper import normalize_quaternion_msg, np_to_kdl, point_to_kdl, kdl_to_point, \
    quaternion_to_kdl, transform_to_kdl, kdl_to_transform_stamped


class DebugTFPublisher(GiskardBehavior):
    """
    Published tf for attached and evironment objects.
    """

    @profile
    def __init__(self, name, tf_topic='/tf'):
        super(DebugTFPublisher, self).__init__(name)
        self.tf_pub = rospy.Publisher(tf_topic, TFMessage, queue_size=10)

    def make_transform(self, parent_frame, child_frame, pose_list):
        q = Quaternion(pose_list[3], pose_list[4], pose_list[5], pose_list[6])
        tf = TransformStamped()
        tf.header.frame_id = parent_frame
        tf.header.stamp = rospy.get_rostime()
        tf.child_frame_id = child_frame
        tf.transform.translation.x = pose_list[0]
        tf.transform.translation.y = pose_list[1]
        tf.transform.translation.z = pose_list[2]
        tf.transform.rotation = normalize_quaternion_msg(q)
        return tf

    def filter_points(self, p_debug, pos_sep='_P_', sep='/'):
        pos_tfs = dict()
        for full_name, d in p_debug.items():
            if pos_sep in full_name:
                b, e = full_name.split(pos_sep)
                f1 = b[len(b) - b[::-1].index(sep):]
                f2 = e[:e.index(sep)]
                k = (f1, f2)
                if k in pos_tfs:
                    pos_tfs[k].append(d)
                else:
                    pos_tfs[k] = [d]
        points = dict()
        for k, vs in pos_tfs.items():
            p = Point()
            p.x = vs[0]
            p.y = vs[1]
            p.z = vs[2]
            points[k] = p
        return points

    def filter_rotations(self, p_debug, rot_sep='_R_', sep='/'):
        rot_tfs = dict()
        for full_name, d in p_debug.items():
            if rot_sep in full_name:
                b, e = full_name.split(rot_sep)
                f1 = b[len(b) - b[::-1].index(sep):]
                f2 = e[:e.index(sep)]
                i, j = map(int, e[e.index(sep)+1:].split(','))
                k = (f1, f2)
                if k in rot_tfs:
                    rot_tfs[k][i][j] = d
                else:
                    rot_tfs[k] = np.zeros((4, 4))
                    rot_tfs[k][i][j] = d
        rotations = dict()
        for k, vs in rot_tfs.items():
            if np.sum(vs[:, -1]) != 1 or np.sum(vs[-1, :]) != 1:
                raise Exception(f'Invalid debug rotation matrix {k[0]}_R_{k[1]}.')
            q = Quaternion(*quaternion_from_matrix(vs))
            rotations[k] = q
        return rotations

    def filter_transforms(self, p_debug, t_sep='_T_', sep='/'):
        tfs = dict()
        for full_name, d in p_debug.items():
            if t_sep in full_name:
                b, e = full_name.split(t_sep)
                f1 = b[len(b) - b[::-1].index(sep):]
                f2 = e[:e.index(sep)]
                i, j = map(int, e[e.index(sep)+1:].split(','))
                k = (f1, f2)
                if k in tfs:
                    tfs[k][i][j] = d
                else:
                    tfs[k] = np.zeros((4, 4))
                    tfs[k][i][j] = d
        transforms = dict()
        for k, vs in tfs.items():
            t = Transform()
            z = np.zeros((4, 4))
            z[:3, :3] = vs[:3, :3]
            z[3, 3] = 1.0
            t.rotation = Quaternion(*quaternion_from_matrix(z))
            t.translation.x = vs[0, 3]
            t.translation.y = vs[1, 3]
            t.translation.z = vs[2, 3]
            transforms[k] = t
        return transforms

    def publish_debug_frames(self, p_debug):
        tf_msg = TFMessage()

        # Add points
        points = self.filter_points(p_debug)
        for k, pos in points.items():
            map_T_k0 = np_to_kdl(self.world.get_fk('map', k[0]))
            k0_P_k1 = point_to_kdl(pos)
            map_P_k1 = kdl_to_point(map_T_k0 * k0_P_k1)
            tf_msg.transforms.append(self.make_transform('map', k[1], [map_P_k1.x,
                                                                       map_P_k1.y,
                                                                       map_P_k1.z,
                                                                       0, 0, 0, 1]))
        # Add rotations
        rotations = self.filter_rotations(p_debug)
        for k, q in rotations.items():
            map_T_k0 = np_to_kdl(self.world.get_fk('map', k[0]))
            k0_T_k0 = np_to_kdl(self.world.get_fk(k[0], k[0]))
            map_P_k1 = kdl_to_transform_stamped(map_T_k0 * k0_T_k0 * quaternion_to_kdl(q), 'map', k[1])
            tf_msg.transforms.append(map_P_k1)

        # Add Transforms
        transforms = self.filter_transforms(p_debug)
        for k, t in transforms.items():
            map_T_k0 = np_to_kdl(self.world.get_fk('map', k[0]))
            k0_T_k1 = transform_to_kdl(t)
            map_P_k1 = kdl_to_transform_stamped(map_T_k0 * k0_T_k1, 'map', k[1])
            tf_msg.transforms.append(map_P_k1)

        self.tf_pub.publish(tf_msg)

    @profile
    def update(self):
        with self.get_god_map() as god_map:
            p_debug = god_map.unsafe_get_data(identifier.debug_expressions_evaluated)
            if len(p_debug.keys()) > 0:
                self.publish_debug_frames(p_debug)
        return Status.RUNNING

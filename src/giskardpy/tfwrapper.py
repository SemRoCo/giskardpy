import tf
import rospy
from geometry_msgs.msg import PoseStamped, Transform, TransformStamped
from tf2_geometry_msgs import do_transform_pose
from tf2_py._tf2 import ExtrapolationException
from tf2_ros import Buffer, TransformListener
from multiprocessing import Lock


class TfWrapper(object):
    def __init__(self, buffer_size=2):
        self.tfBuffer = Buffer(rospy.Duration(buffer_size))
        self.tf_listener = TransformListener(self.tfBuffer)
        self.tf_frequency = rospy.Duration(1.0)
        self.broadcasting_frames = []
        self.broadcasting_frames_lock = Lock()
        rospy.sleep(0.1)

    def transform_pose(self, target_frame, pose):
        try:
            transform = self.tfBuffer.lookup_transform(target_frame,
                                                       pose.header.frame_id,  # source frame
                                                       pose.header.stamp,
                                                       rospy.Duration(1.0))
            new_pose = do_transform_pose(pose, transform)
            return new_pose
        except ExtrapolationException as e:
            rospy.logwarn(e)

    def lookup_transform(self, target_frame, source_frame):
        p = PoseStamped()
        p.header.frame_id = source_frame
        p.pose.orientation.w = 1.0
        return self.transform_pose(target_frame, p)

    def add_frame_from_pose(self, name, pose_stamped):
        with self.broadcasting_frames_lock:
            frame = TransformStamped()
            frame.header = pose_stamped.header
            frame.child_frame_id = name
            frame.transform.translation = pose_stamped.pose.position
            frame.transform.rotation = pose_stamped.pose.orientation
            self.broadcasting_frames.append(frame)

    def start_frame_broadcasting(self):
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_timer = rospy.Timer(self.tf_frequency, self.broadcasting_cb)

    def broadcasting_cb(self, data):
        with self.broadcasting_frames_lock:
            for frame in self.broadcasting_frames:
                self.tf_broadcaster.sendTransformMessage(frame)

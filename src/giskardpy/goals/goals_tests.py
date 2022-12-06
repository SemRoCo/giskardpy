import numpy as np
from geometry_msgs.msg import QuaternionStamped, PointStamped, PoseStamped, Vector3Stamped
import giskardpy.casadi_wrapper as w
from giskardpy.goals.goal import Goal


class DebugGoal(Goal):
    def __init__(self):
        super().__init__()

    def make_constraints(self):
        q = QuaternionStamped()
        q.header.frame_id = self.world.root_link_name
        q.quaternion.w = 1
        q = w.Quaternion(q)
        self.add_debug_expr('q', q)

        p = PointStamped()
        p.header.frame_id = self.world.root_link_name
        p.point.x = 1
        p = w.Point3(p)
        self.add_debug_expr('p', p)

        pose = PoseStamped()
        pose.header.frame_id = self.world.root_link_name
        pose.pose.position.y = 1
        pose.pose.orientation.w = 1
        pose = w.TransMatrix(pose)
        self.add_debug_expr('pose', pose)

        v = Vector3Stamped()
        v.header.frame_id = self.world.root_link_name
        v.vector.x = 1
        v = w.Vector3(v)
        self.add_debug_expr('v', v)

        r = QuaternionStamped()
        r.header.frame_id = self.world.root_link_name
        r.quaternion.w = 1
        r = w.RotationMatrix(r)
        self.add_debug_expr('r', r)

        e1 = w.Expression(np.eye(3))
        self.add_debug_expr('e1', e1)

        e2 = w.Expression(np.array([1, 2, 3]))
        self.add_debug_expr('e2', e2)

        s = self.get_sampling_period_symbol()
        self.add_debug_expr('s', s)

        self.add_debug_expr('f', 23)

    def __str__(self) -> str:
        return super().__str__()

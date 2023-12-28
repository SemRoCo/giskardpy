from typing import Optional, List

import numpy as np
from geometry_msgs.msg import QuaternionStamped, PointStamped, PoseStamped, Vector3Stamped
import giskardpy.casadi_wrapper as w
from giskardpy.goals.goal import Goal, NonMotionGoal
from giskardpy.goals.monitors.monitors import ExpressionMonitor
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager


class DebugGoal(NonMotionGoal):
    def __init__(self,
                 name: Optional[str] = None,
                 start_monitors: Optional[List[ExpressionMonitor]] = None,
                 hold_monitors: Optional[List[ExpressionMonitor]] = None,
                 end_monitors: Optional[List[ExpressionMonitor]] = None):
        if name is None:
            name = self.__class__.__name__
        super().__init__(name=name)
        q = QuaternionStamped()
        q.header.frame_id = god_map.world.root_link_name
        q.quaternion.w = 1
        q = w.Quaternion(q)
        god_map.debug_expression_manager.add_debug_expression('q', q)

        p = PointStamped()
        p.header.frame_id = god_map.world.root_link_name
        p.point.x = 1
        p = w.Point3(p)
        god_map.debug_expression_manager.add_debug_expression('p', p)

        pose = PoseStamped()
        pose.header.frame_id = god_map.world.root_link_name
        pose.pose.position.y = 1
        pose.pose.orientation.w = 1
        pose = w.TransMatrix(pose)
        god_map.debug_expression_manager.add_debug_expression('pose', pose)

        v = Vector3Stamped()
        v.header.frame_id = god_map.world.root_link_name
        v.vector.x = 1
        v = w.Vector3(v)
        god_map.debug_expression_manager.add_debug_expression('v', v)

        r = QuaternionStamped()
        r.header.frame_id = god_map.world.root_link_name
        r.quaternion.w = 1
        r = w.RotationMatrix(r)
        god_map.debug_expression_manager.add_debug_expression('r', r)

        e1 = w.Expression(np.eye(3))
        god_map.debug_expression_manager.add_debug_expression('e1', e1)

        e2 = w.Expression(np.array([1, 2, 3]))
        god_map.debug_expression_manager.add_debug_expression('e2', e2)

        t = symbol_manager.time
        god_map.debug_expression_manager.add_debug_expression('t', t)

        god_map.debug_expression_manager.add_debug_expression('f', 23)

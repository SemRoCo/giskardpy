from typing import Optional, List

import numpy as np
from geometry_msgs.msg import QuaternionStamped, PointStamped, PoseStamped, Vector3Stamped
import giskardpy.casadi_wrapper as cas
from giskardpy.goals.goal import NonMotionGoal, Goal
from giskardpy.monitors.monitors import ExpressionMonitor
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager


class DebugGoal(NonMotionGoal):
    def __init__(self,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
        if name is None:
            name = self.__class__.__name__
        super().__init__(name=name)
        q = QuaternionStamped()
        q.header.frame_id = god_map.world.root_link_name
        q.quaternion.w = 1
        q = cas.Quaternion(q)
        god_map.debug_expression_manager.add_debug_expression('q', q)

        p = PointStamped()
        p.header.frame_id = god_map.world.root_link_name
        p.point.x = 1
        p = cas.Point3(p)
        god_map.debug_expression_manager.add_debug_expression('p', p)

        pose = PoseStamped()
        pose.header.frame_id = god_map.world.root_link_name
        pose.pose.position.y = 1
        pose.pose.orientation.w = 1
        pose = cas.TransMatrix(pose)
        god_map.debug_expression_manager.add_debug_expression('pose', pose)

        v = Vector3Stamped()
        v.header.frame_id = god_map.world.root_link_name
        v.vector.x = 1
        v = cas.Vector3(v)
        god_map.debug_expression_manager.add_debug_expression('v', v)

        r = QuaternionStamped()
        r.header.frame_id = god_map.world.root_link_name
        r.quaternion.w = 1
        r = cas.RotationMatrix(r)
        god_map.debug_expression_manager.add_debug_expression('r', r)

        e1 = cas.Expression(np.eye(3))
        god_map.debug_expression_manager.add_debug_expression('e1', e1)

        e2 = cas.Expression(np.array([1, 2, 3]))
        god_map.debug_expression_manager.add_debug_expression('e2', e2)

        t = symbol_manager.time
        god_map.debug_expression_manager.add_debug_expression('t', t)

        god_map.debug_expression_manager.add_debug_expression('f', 23)


class CannotResolveSymbol(Goal):

    def __init__(self, name: str, joint_name: str, start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol, end_condition: cas.Expression = cas.TrueSymbol):
        super().__init__(name=name, start_condition=start_condition, hold_condition=hold_condition,
                         end_condition=end_condition)
        self.data = {}
        s = self.get_symbol_for_self_attribute('.data[2]')
        t = self.create_and_add_task('muh')
        joint_name = god_map.world.search_for_joint_name(joint_name)
        joint_position = self.get_joint_position_symbol(joint_name)
        t.add_equality_constraint(reference_velocity=1,
                                  equality_bound=1,
                                  weight=1,
                                  task_expression=s * joint_position)

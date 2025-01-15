from typing import Optional

import numpy as np
import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import Derivatives
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.tasks.task import Task
from giskardpy.symbol_manager import symbol_manager


class DebugGoal(Task):
    def __init__(self,
                 name: Optional[str] = None):
        if name is None:
            name = self.__class__.__name__
        super().__init__(name=name)
        q = cas.Quaternion(reference_frame=god_map.world.root_link_name)
        god_map.debug_expression_manager.add_debug_expression('q', q)

        p = cas.Point3((1, 0, 0), reference_frame=god_map.world.root_link_name)
        god_map.debug_expression_manager.add_debug_expression('p', p)

        pose = cas.TransMatrix.from_xyz_rpy(y=1, reference_frame=god_map.world.root_link_name)
        god_map.debug_expression_manager.add_debug_expression('pose', pose)

        v = cas.Vector3((1, 0, 0), reference_frame=god_map.world.root_link_name)
        god_map.debug_expression_manager.add_debug_expression('v', v)

        r = cas.Quaternion(reference_frame=god_map.world.root_link_name).to_rotation_matrix()
        god_map.debug_expression_manager.add_debug_expression('r', r)

        e1 = cas.Expression(np.eye(3))
        god_map.debug_expression_manager.add_debug_expression('e1', e1)

        e2 = cas.Expression(np.array([1, 2, 3]))
        god_map.debug_expression_manager.add_debug_expression('e2', e2)

        t = symbol_manager.time
        god_map.debug_expression_manager.add_debug_expression('t', t)

        god_map.debug_expression_manager.add_debug_expression('f', 23)


class CannotResolveSymbol(Task):

    def __init__(self, name: str, joint_name: str):
        super().__init__(name=name)
        self.data = {}
        s = symbol_manager.get_symbol(self.ref_str + '.data[2]')
        joint_name = god_map.world.search_for_joint_name(joint_name)
        joint = god_map.world.joints[joint_name]
        joint_position = joint.get_symbol(Derivatives.position)
        self.add_equality_constraint(reference_velocity=1,
                                  equality_bound=1,
                                  weight=1,
                                  task_expression=s * joint_position)

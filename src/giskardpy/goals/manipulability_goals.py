from typing import Optional, List

from giskardpy import casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.monitors.monitors import ExpressionMonitor
from giskardpy.tasks.task import Task
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager


class MaxManipulability(Goal):
    def __init__(self, root_link: str, tip_link: str, gain: float = 0.5,
                 name: Optional[str] = None,
                 optimize_rotational_dofs=False,
                 monitor_threshold: float = 0.01,
                 prediction_horizon: int = 5,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
        self.root_link = god_map.world.search_for_link_name(root_link, None)
        self.tip_link = god_map.world.search_for_link_name(tip_link, None)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)

        results = god_map.world.compute_split_chain(self.root_link, self.tip_link, True, True, False, False)
        if len(results[0]) > 0 and len(results[1]) > 0 and len(results[2]) > 0:
            raise Exception('tip link and root link are in different branches of the kinematic chain of Maximize Manipulability Goal')
        elif len(results[0]) > 0:
            raise Exception('tip link is below root link in kinematic chain of Maximize Manipulability Goal')
        for joint in results[2]:
            if 'joint' in joint and not god_map.world.is_joint_rotational(joint):
                raise Exception('Non rotational joint in kinematic chain of Maximize Manipulability Goal')

        task = self.create_and_add_task('manipulability')
        root_T_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link)
        task.add_manipulability_constraint_vector(names=['x', 'y', 'z'],
                                                  task_expressions=root_T_tip.to_position()[:3],
                                                  gain=gain,
                                                  prediction_horizon=prediction_horizon)
        if optimize_rotational_dofs:
            task.add_manipulability_constraint_vector(names=['qx', 'qy', 'qz'],
                                                      task_expressions=root_T_tip.to_rotation().to_quaternion()[:3],
                                                      gain=gain,
                                                      prediction_horizon=prediction_horizon)
        m = symbol_manager.get_symbol(f'god_map.qp_controller.manipulability_indexes[0]')
        old_m = symbol_manager.get_symbol(f'god_map.qp_controller.manipulability_indexes[1]')
        god_map.debug_expression_manager.add_debug_expression('mIndex_percentualDifference',
                                                              1 - cas.min(cas.save_division(old_m, m), 1))
        percentual_diff = 1 - cas.min(cas.save_division(old_m, m), 1)
        monitor = ExpressionMonitor(name=f'manipMonitor{tip_link}')
        self.add_monitor(monitor)
        monitor.set_expression(cas.less(percentual_diff, monitor_threshold))
        task.end_condition = monitor

    """
    This goal maximizes the manipulability of the kinematic chain between root_link and tip_link.
    This chain should only include rotational joint and no linear joints i.e. torso lift joints or odometry joints.
    The expressions this goal writes onto the god_map will be used in the qp building logic to adapt the linear weights 
    of the QP. To be more precise, first the Jacobian of the expressions is calculated and then the gradient of 
    the jacobian is calculated. The jacobian is then evaluated and used to calculate the manipulability index
    m=sqrt(det(JJ^T)). The index is then multiplied with the gradient of J. Finally, the linear weight is calculated by 
    multiplying this with a negative gain value.
    Future work combines translation and rotation, and improves the monitoring to not stop to early during the main motion.
    """

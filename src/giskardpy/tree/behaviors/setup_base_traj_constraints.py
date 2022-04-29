from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy import casadi_wrapper as w
from giskardpy.exceptions import GiskardException
from giskardpy.goals.base_traj_follower import BaseTrajFollower
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.logging import loginfo
from giskardpy.utils.utils import convert_dictionary_to_ros_message, catch_and_raise_to_blackboard


class SetupBaseTrajConstraints(GiskardBehavior):
    @profile
    @catch_and_raise_to_blackboard
    def update(self):
        # TODO make this interruptable
        loginfo('setting up base traj constraints.')
        self.get_god_map().set_data(identifier.goals, {})

        self.soft_constraints = {}
        self.vel_constraints = {}
        self.debug_expr = {}

        c = BaseTrajFollower(god_map=self.god_map, joint_name='brumbrum')
        soft_constraints, vel_constraints, debug_expressions = c.get_constraints()
        self.soft_constraints.update(soft_constraints)
        self.vel_constraints.update(vel_constraints)
        self.debug_expr.update(debug_expressions)

        self.get_god_map().set_data(identifier.constraints, self.soft_constraints)
        self.get_god_map().set_data(identifier.vel_constraints, self.vel_constraints)
        if len(self.soft_constraints) == 0 and len(self.vel_constraints) == 0:
            raise GiskardException('Goals parsing resulted in no soft or velocity constraints')
        self.get_god_map().set_data(identifier.debug_expressions, self.debug_expr)

        if self.get_god_map().get_data(identifier.check_reachability):
            raise NotImplementedError('reachability check is not implemented')

        l = self.active_free_symbols()
        free_variables = list(sorted([v for v in self.world.joint_constraints if v.position_name in l],
                                     key=lambda x: x.position_name))
        if len(free_variables) == 0:
            raise GiskardException('Goal parsing resulted in no free variables.')
        self.get_god_map().set_data(identifier.free_variables, free_variables)
        loginfo('Done parsing traj message.')
        return Status.SUCCESS

    def active_free_symbols(self):
        symbols = set()
        for c in self.soft_constraints.values():
            symbols.update(str(s) for s in w.free_symbols(c.expression))
        return symbols

    def replace_jsons_with_ros_messages(self, d):
        for key, value in d.items():
            if isinstance(value, dict) and 'message_type' in value:
                d[key] = convert_dictionary_to_ros_message(value)
        return d

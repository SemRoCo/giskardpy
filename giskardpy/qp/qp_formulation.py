from enum import IntEnum


class QPFormulation(IntEnum):
    no_mpc = -1
    explicit = 0
    implicit = 1
    explicit_no_acc = 2
    explicit_explicit_pos_limits = 10
    implicit_explicit_pos_limits = 11
    explicit_no_acc_explicit_pos_limits = 12
    implicit_variable_dt = 21

    def explicit_pos_limits(self) -> bool:
        return 20 > self > 10

    def is_dt_variable(self) -> bool:
        return self > 20

    def is_no_mpc(self) -> bool:
        return self == self.no_mpc

    def has_acc_variables(self) -> bool:
        return self.is_explicit()

    def has_jerk_variables(self) -> bool:
        return self.is_explicit() or self.is_explicit_no_acc()

    def is_mpc(self) -> bool:
        return not self.is_no_mpc()

    def is_implicit(self) -> bool:
        return self in [self.implicit, self.implicit_explicit_pos_limits]

    def is_explicit(self) -> bool:
        return self in [self.explicit, self.explicit_explicit_pos_limits]

    def is_explicit_no_acc(self) -> bool:
        return self in [self.explicit_no_acc, self.explicit_no_acc_explicit_pos_limits]
from dataclasses import dataclass, field


@dataclass
class QPFormulation:
    double_qp: bool = field(default=False)
    is_mpc: bool = field(default=True)
    has_explicit_pos_limits: bool = field(default=False)
    has_explicit_acc_variables: bool = field(default=False)
    has_explicit_jerk_variables: bool = field(default=True)

    @property
    def is_implicit(self) -> bool:
        return self.is_mpc and not self.has_explicit_acc_variables and not self.has_explicit_jerk_variables

    @property
    def is_explicit(self) -> bool:
        return self.is_mpc and self.has_explicit_acc_variables and self.has_explicit_jerk_variables

    @classmethod
    def Implicit(cls):
        return cls(is_mpc=True,
                   has_explicit_pos_limits=False,
                   has_explicit_acc_variables=False,
                   has_explicit_jerk_variables=False)

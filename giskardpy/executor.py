from dataclasses import dataclass, field, InitVar

from typing_extensions import Optional

from giskardpy.data_types.exceptions import (
    NoQPControllerConfigException,
)
from giskardpy.model.better_pybullet_syncer import BulletCollisionDetector
from giskardpy.model.collision_world_syncer import (
    CollisionWorldSynchronizer,
    CollisionCheckerLib,
)
from giskardpy.model.collisions import NullCollisionDetector
from giskardpy.motion_statechart.auxilary_variable_manager import (
    AuxiliaryVariableManager,
    AuxiliaryVariable,
)
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.qp.exceptions import EmptyProblemException
from giskardpy.qp.qp_controller import QPController
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import (
    AbstractRobot,
)
from semantic_digital_twin.world import World


@dataclass
class Executor:
    """
    Represents the main execution entity that manages motion statecharts, collision
    scenes, and control cycles for the robot's operations.

    :ivar tmp_folder: Temporary folder path used for auxiliary operations during execution.
    :type tmp_folder: str
    :ivar motion_statechart: The motion statechart describing the robot's motion logic.
    :type motion_statechart: MotionStatechart
    :ivar collision_scene: The collision scene synchronizer for managing robot collision states.
    :type collision_scene: Optional[CollisionWorldSynchronizer]
    :ivar auxiliary_variable_manager: Manages auxiliary symbolic variables for execution contexts.
    :type auxiliary_variable_manager: AuxiliaryVariableManager
    :ivar qp_controller: Optional quadratic programming controller used for motion control.
    :type qp_controller: Optional[QPController]
    :ivar control_cycles: Tracks the number of control cycles elapsed during execution.
    :type control_cycles: int
    """

    world: World
    """The world object containing the state and entities of the robot's environment."""
    controller_config: Optional[QPControllerConfig] = None
    """Optional configuration for the QP Controller. Is only needed when constraints are present in the motion statechart."""
    collision_checker: InitVar[CollisionCheckerLib] = field(
        default=CollisionCheckerLib.none
    )
    """Library used for collision checking. Can be set to Bullet or None."""
    tmp_folder: str = field(default="/tmp/")
    """Path to safe temporary files."""

    # %% init False
    motion_statechart: MotionStatechart = field(init=False)
    """The motion statechart describing the robot's motion logic."""
    collision_scene: Optional[CollisionWorldSynchronizer] = field(
        default=None, init=False
    )
    """The collision scene synchronizer for managing robot collision states."""
    auxiliary_variable_manager: AuxiliaryVariableManager = field(
        default_factory=AuxiliaryVariableManager, init=False
    )
    """Manages auxiliary symbolic variables for execution contexts."""
    qp_controller: Optional[QPController] = field(default=None, init=False)
    """Optional quadratic programming controller used for motion control."""

    control_cycles: int = field(init=False)
    """Tracks the number of control cycles elapsed during execution."""
    _control_cycles_variable: AuxiliaryVariable = field(init=False)
    """Auxiliary variable linked to the control_cycles attribute."""

    _time: float = field(init=False)
    """The time that has passed since the execution started."""
    _time_variable: AuxiliaryVariable = field(init=False)
    """Auxiliary variable representing the current time in seconds since the start of the simulation."""

    def __post_init__(self, collision_checker: CollisionCheckerLib):
        if collision_checker == CollisionCheckerLib.bpb:
            collision_detector = BulletCollisionDetector(
                _world=self.world, tmp_folder=self.tmp_folder
            )
        else:
            collision_detector = NullCollisionDetector(_world=self.world)

        self.collision_scene = CollisionWorldSynchronizer(
            world=self.world,
            robots=self.world.get_semantic_annotations_by_type(AbstractRobot),
            collision_detector=collision_detector,
        )

    def _create_control_cycles_variable(self):
        self._control_cycles_variable = (
            self.auxiliary_variable_manager.create_float_variable(
                PrefixedName("control_cycles"), lambda: self.control_cycles
            )
        )

    def compile(self, motion_statechart: MotionStatechart):
        self.motion_statechart = motion_statechart
        self.control_cycles = 0
        self._create_control_cycles_variable()
        self.motion_statechart.compile(self.build_context)
        self._compile_qp_controller(self.controller_config)

    @property
    def build_context(self) -> BuildContext:
        return BuildContext(
            world=self.world,
            auxiliary_variable_manager=self.auxiliary_variable_manager,
            collision_scene=self.collision_scene,
            qp_controller_config=self.controller_config,
            control_cycle_variable=self._control_cycles_variable,
        )

    def tick(self):
        self.control_cycles += 1
        self.collision_scene.sync()
        self.collision_scene.check_collisions()
        self.motion_statechart.tick(self.build_context)
        if self.qp_controller is None:
            return
        next_cmd = self.qp_controller.get_cmd(
            world_state=self.world.state.data,
            life_cycle_state=self.motion_statechart.life_cycle_state.data,
            external_collisions=self.collision_scene.get_external_collision_data(),
            self_collisions=self.collision_scene.get_self_collision_data(),
            auxiliary_variables=self.auxiliary_variable_manager.resolve_auxiliary_variables(),
        )
        self.world.apply_control_commands(
            next_cmd,
            self.qp_controller.config.control_dt or self.qp_controller.config.mpc_dt,
            self.qp_controller.config.max_derivative,
        )

    def tick_until_end(self, timeout: int = 1_000):
        """
        Calls tick until is_end_motion() returns True.
        :param timeout: Max number of ticks to perform.
        """
        for i in range(timeout):
            self.tick()
            if self.motion_statechart.is_end_motion():
                return
        raise TimeoutError("Timeout reached while waiting for end of motion.")

    def _compile_qp_controller(self, controller_config: QPControllerConfig):
        ordered_dofs = sorted(
            self.world.active_degrees_of_freedom,
            key=lambda dof: self.world.state._index[dof.name],
        )
        constraint_collection = (
            self.motion_statechart.combine_constraint_collections_of_nodes()
        )
        if len(constraint_collection.constraints) == 0:
            self.qp_controller = None
            # to not build controller, if there are no constraints
            return
        elif controller_config is None:
            raise NoQPControllerConfigException(
                "constraints but no controller config given."
            )
        self.qp_controller = QPController(
            config=controller_config,
            degrees_of_freedom=ordered_dofs,
            constraint_collection=constraint_collection,
            world_state_symbols=self.world.state.get_variables(),
            life_cycle_variables=self.motion_statechart.life_cycle_state.life_cycle_symbols(),
            external_collision_avoidance_variables=self.collision_scene.get_external_collision_symbol(),
            self_collision_avoidance_variables=self.collision_scene.get_self_collision_symbol(),
            auxiliary_variables=self.auxiliary_variable_manager.variables,
        )
        if self.qp_controller.has_not_free_variables():
            raise EmptyProblemException()

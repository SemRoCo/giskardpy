from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import (
    List,
    Dict,
    Optional,
    TYPE_CHECKING,
    Literal,
    Self,
)

from giskardpy.motion_statechart.data_types import TransitionKind
from giskardpy.motion_statechart.plotters.styles import (
    ConditionColors,
    MonitorStyle,
    MonitorShape,
    TaskShape,
    TaskStyle,
    GoalNodeStyle,
    GoalNodeShape,
)

if TYPE_CHECKING:
    pass


@dataclass
class NodePlotSpec:
    visible: bool = True
    style: str = "filled, rounded"
    shape: str = "rectangle"
    extra_border_styles: List[str] = field(default_factory=list)

    @classmethod
    def create_monitor_style(cls) -> Self:
        return cls(
            visible=True, style=MonitorStyle, shape=MonitorShape, extra_border_styles=[]
        )

    @classmethod
    def create_task_style(cls) -> Self:
        return cls(
            visible=True, style=TaskStyle, shape=TaskShape, extra_border_styles=[]
        )

    @classmethod
    def create_goal_style(cls) -> Self:
        return cls(
            visible=True,
            style=GoalNodeStyle,
            shape=GoalNodeShape,
            extra_border_styles=[],
        )

    @classmethod
    def create_end_style(cls):
        return cls(
            visible=True,
            style=MonitorStyle,
            shape=MonitorShape,
            extra_border_styles=["rounded"],
        )

    @classmethod
    def create_cancel_style(cls):
        return cls(
            visible=True,
            style=MonitorStyle,
            shape=MonitorShape,
            extra_border_styles=["dashed, rounded"],
        )


SrcSelector = Literal["parent", "child"]
DstSelector = Literal["parent", "child"]
StateSelector = Literal["parent", "child"]


@dataclass(frozen=True)
class EdgeSpec:
    color: str
    src_selector: SrcSelector
    dst_selector: DstSelector
    state_selector: StateSelector
    extra_edge_kwargs: Optional[Dict[str, object]] = None

    def extras(self) -> Dict[str, object]:
        return {} if self.extra_edge_kwargs is None else dict(self.extra_edge_kwargs)


TRANSITION_SPECS: Dict[TransitionKind, EdgeSpec] = {
    TransitionKind.START: EdgeSpec(
        color=ConditionColors.StartCondColor,
        src_selector="child",
        dst_selector="parent",
        state_selector="parent",
    ),
    TransitionKind.PAUSE: EdgeSpec(
        color=ConditionColors.PauseCondColor,
        src_selector="child",
        dst_selector="parent",
        state_selector="child",
        extra_edge_kwargs={"minlen": 0},
    ),
    TransitionKind.END: EdgeSpec(
        color=ConditionColors.EndCondColor,
        src_selector="child",
        dst_selector="parent",
        state_selector="child",
        extra_edge_kwargs={
            "arrowhead": "none",
            "arrowtail": "normal",
            "dir": "both",
        },
    ),
    TransitionKind.RESET: EdgeSpec(
        color=ConditionColors.ResetCondColor,
        src_selector="parent",
        dst_selector="child",
        state_selector="parent",
        extra_edge_kwargs={
            "arrowhead": "none",
            "arrowtail": "normal",
            "dir": "both",
            "minlen": 0,
        },
    ),
}

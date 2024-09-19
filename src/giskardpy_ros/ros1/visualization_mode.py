from enum import Enum


class VisualizationMode(Enum):
    Nothing = -1
    Visuals = 0
    Collisions = 1
    CollisionsDecomposed = 2
    VisualsFrameLocked = 3
    CollisionsFrameLocked = 4
    CollisionsDecomposedFrameLocked = 5

    def is_visual(self) -> bool:
        return self in VISUAL_MODES

    def is_original_collision(self) -> bool:
        return self in COLLISION_MODES

    def is_collision_decomposed(self) -> bool:
        return self in COLLISION_DECOMPOSED_MODES

    def is_frame_locked(self) -> bool:
        return self in FRAME_LOCKED_MODES


VISUAL_MODES = {VisualizationMode.Visuals,
                VisualizationMode.VisualsFrameLocked}
COLLISION_MODES = {VisualizationMode.Collisions,
                   VisualizationMode.CollisionsDecomposed}
COLLISION_DECOMPOSED_MODES = {VisualizationMode.CollisionsDecomposed,
                              VisualizationMode.CollisionsDecomposedFrameLocked}
FRAME_LOCKED_MODES = {VisualizationMode.VisualsFrameLocked,
                      VisualizationMode.CollisionsFrameLocked,
                      VisualizationMode.CollisionsDecomposedFrameLocked}

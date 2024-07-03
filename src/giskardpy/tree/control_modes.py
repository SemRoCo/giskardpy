from enum import Enum


class ControlModes(Enum):
    none = -1
    open_loop = 1
    close_loop = 2
    standalone = 3
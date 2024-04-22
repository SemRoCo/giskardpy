from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from enum import IntEnum
from typing import Optional, Generic, TypeVar, Dict, Union

import genpy
import numpy as np
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped, QuaternionStamped
from sensor_msgs.msg import JointState
import std_msgs.msg as std_msgs

class GiskardErrorCodes(IntEnum):
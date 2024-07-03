from typing import Dict, Tuple, List

from giskard_msgs.msg import CollisionEntry
from giskardpy.data_types import PrefixName


# class CollisionMatrix:
#     matrix: Dict[Tuple[PrefixName, PrefixName], float]
#
#     def __init__(self):
#         self.matrix = {}
#
#     @classmethod
#     def from_collision_entries(cls, collision_entires: List[CollisionEntry]):
#         collision_check_distances = self.create_collision_check_distances()
#         # ignored_collisions = god_map.get_collision_scene().ignored_self_collion_pairs
#         collision_matrix = god_map.collision_scene.collision_goals_to_collision_matrix(deepcopy(collision_entries),
#                                                                                        collision_check_distances)
#         return collision_matrix
#
from giskardpy import identifier
from giskardpy.god_map import GodMap


class TimeCollector:
    constraints = []
    variables = []
    lengths = []
    times = []
    jacobians = []
    compilations = []
    collision_avoidance = []

    def __init__(self, god_map: GodMap):
        self.god_map = god_map

    def next_goal(self):
        pass

    def print(self):
        robot_name = list(self.god_map.unsafe_get_data(identifier.world).groups.keys())[0]
        print()
        for constraint, variable, length, time, jacobian, compilation, collision_avoidance in zip(self.constraints,
                                                                                                  self.variables,
                                                                                                  self.lengths,
                                                                                                  self.times,
                                                                                                  self.jacobians,
                                                                                                  self.compilations,
                                                                                                  self.collision_avoidance):
            print(f'{robot_name};{constraint};{variable};{length};{time};{jacobian};{compilation};{collision_avoidance}')
        print()
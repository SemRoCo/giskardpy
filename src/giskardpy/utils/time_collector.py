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

    def __init__(self):
        self.god_map = GodMap()

    def next_goal(self):
        pass

    def pretty_print(self, filter=None):
        if filter is None:
            filter = lambda i, x: bool
        s = f'constraints: {[x for i, x in enumerate(self.constraints) if filter(i, x)]}\n' \
            f'variables: {[x for i, x in enumerate(self.variables) if filter(i, x)]}\n' \
            f'length: {[x for i, x in enumerate(self.lengths) if filter(i, x)]}\n' \
            f'times: {[x for i, x in enumerate(self.times) if filter(i, x)]}\n' \
            f'jacobians: {[x for i, x in enumerate(self.jacobians) if filter(i, x)]}\n' \
            f'compilation: {[x for i, x in enumerate(self.compilations) if filter(i, x)]}\n' \
            f'colllision_avoidance: {[x for i, x in enumerate(self.collision_avoidance) if filter(i, x)]}'
        print(s)

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
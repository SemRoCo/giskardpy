from giskardpy.god_map import GodMap
from smach import State

# class State(object):
#     def __init__(self, outcomes=()):
#         self.plugins = {}
#         self.god_map = None
#
#     def execute(self, previous_god_map):
#         pass
#
#     def get_god_map_copy(self):
#         return self.god_map
#
#     def get_god_map(self):
#         return self.god_map


class Monitor(State):
    def __init__(self, outcomes=('start_planning', 'end')):
        super(Monitor, self).__init__(outcomes)

    def execute(self, previous_god_map):
        # while no goal received:
        for plugin in self.plugins.values():
            plugin.update()

        # if goal received:
        return 'start_planning'

class InitGodMap(State):
    def __init__(self, outcomes=(), plugins=()):
        super(InitGodMap, self).__init__(outcomes)


class Planning(State):
    def __init__(self, outcomes=('constraints_satisfied', 'exception')):
        super(Planning, self).__init__(outcomes)

    def execute(self, previous_god_map):
        #while planning not finished
        for plugin in self.plugins.values():
            plugin.update()


class StateManager(object):
    def __init__(self):
        self.active_state = None
        self.states = {}
        self.transition_function = {}

    def add_transition(self, name, state_class, transition):
        self.states[name] = state_class
        self.transition_function = transition

    def loop(self):
        last_gm = GodMap()
        while True:
            result = self.active_state.execute(last_gm)
            last_gm = self.active_state.get_god_map()
            self.state_transition(result)

    def state_transition(self, result):
        self.active_state = self.states[self.transition_function[result]]

if __name__ == u'__main__':
    sm = StateManager()
    sm.add_transition('init', InitGodMap(), {'next', 'monitor'})
    sm.add_transition('monitor', Planning(), {'start_planning': 'planning'})
    sm.add_transition('planning', Planning(), {'constraints_satisfied': 'monitor',
                                               'exception': 'monitor'})
    sm.loop()
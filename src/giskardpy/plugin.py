from copy import deepcopy


class Plugin(object):
    # TODO do we even need normal plugins?
    def start(self, god_map):
        self.god_map = god_map

    def stop(self):
        raise NotImplementedError('Please implement the stop method of this plugin.')

    def create_parallel_universe(self):
        return False

    def end_parallel_universe(self):
        return False

    def post_mortem_analysis(self, god_map):
        pass

    def copy(self):
        """
        create copy for parallel universe. make sure to have no links to real robot
        """
        return deepcopy(self)

class IOPlugin(Plugin):
    # TODO merge update and get_readings?
    def get_readings(self):
        return {}

    def update(self):
        pass


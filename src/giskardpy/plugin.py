class Plugin(object):
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
        return self.__class__()

    def get_readings(self):
        return {}

    def update(self):
        pass


class Plugin(object):
    def start(self, god_map):
        self.god_map = god_map

    def stop(self):
        # TODO 2 stops? 1 for main loop, 1 for parallel shit
        pass

    def update(self):
        pass

    def create_parallel_universe(self):
        return False

    def end_parallel_universe(self):
        return False

    def post_mortem_analysis(self, god_map):
        """
        Analyse the memory of a dead god from a parallel universe.
        :param god_map: the dead gods memory
        :type: GodMap
        """
        pass

    def get_readings(self):
        return {}

    def get_replacement_parallel_universe(self):
        return self.__class__()

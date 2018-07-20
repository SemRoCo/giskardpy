class Plugin(object):
    def __init__(self):
        self.started = False

    def start(self, god_map):
        """
        :param god_map:
        :type god_map: giskardpy.god_map.GodMap
        :return:
        """
        self.god_map = god_map
        if not self.started:
            self.start_once()
            self.started = True
        self.start_always()

    def start_once(self):
        pass

    def start_always(self):
        pass

    def stop(self):
        # TODO 2 stops? 1 for main loop, 1 for parallel shit
        pass

    def update(self):
        pass

    def create_parallel_universe(self):
        return False

    def end_parallel_universe(self):
        return False

    def post_mortem_analysis(self, god_map, exception):
        """
        Analyse the memory of a dead god from a parallel universe.
        :param god_map: the dead gods memory
        :type god_map: GodMap
        :param exception:
        :type exception: Exception
        """
        pass

    def get_replacement(self):
        """
        :return:
        :rtype: Plugin
        """
        c = self.copy()
        c.started = self.started
        return c

    def copy(self):
        """
        :return:
        :rtype: Plugin
        """
        raise(NotImplementedError)



class PluginContainer(Plugin):
    def __init__(self, replacement, call_start=False):
        """
        :param replacement:
        :type replacement: Plugin
        :param call_start:
        :type call_start: bool
        """
        self.replacement = replacement
        self.call_init = call_start
        super(PluginContainer, self).__init__()

    def copy(self):
        return self.replacement

    def start(self, god_map):
        if self.call_init:
            self.replacement.start(god_map)

    def get_replacement(self):
        c = self.copy()
        return c
from time import time

from giskardpy.application import ROSApplication
from giskardpy.plugin import PluginBase, PluginParallelUniverseOnly
from giskardpy.process_manager import ProcessManager
from ros_trajectory_controller_main import giskard_pm


class Success(PluginBase):

    # def update(self):
    #     print('.')

    def copy(self):
        return self


class Counter(PluginBase):
    def __init__(self):
        self.count = 0
        super(Counter, self).__init__()

    def update(self):
        self.count += 1
        if self.count == 10000 or self.count == 1000 or self.count == 100000:
            print('{} {}'.format(self.count, (time() - self.god_map.safe_get_data(['time'])) / self.count))


    def end_parallel_universe(self):
        return True

class StartParallel(PluginBase):

    def __init__(self, create=True):
        self.create = create
        super(StartParallel, self).__init__()

    def create_parallel_universe(self):
        if self.create:
            return True
        else:
            self.create = True
            return False

    def copy(self):
        return Success()

    def post_mortem_analysis(self, god_map, exception):
        self.create = False


pm = ProcessManager()
pm.get_god_map().safe_set_data(['time'], time())
pm.register_plugin(u'init pb',Success())
pm.register_plugin(u'urdf',Success())

pm.register_plugin(u'has_goal',StartParallel())
pm.register_plugin(u'js',Success())
pm.register_plugin(u'pw',Success())
pm.register_plugin(u'fk',Success())
pm.register_plugin(u'as',Success())

pm.register_plugin(u'log',PluginParallelUniverseOnly(Counter()))


app = ROSApplication(pm)
app.run()
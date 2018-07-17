import traceback
from collections import OrderedDict
from copy import copy
from time import sleep, time

import rospy

from giskardpy.god_map import GodMap
from giskardpy.exceptions import NameConflictException, MAX_NWSR_REACHEDException, QPSolverException


class ProcessManager(object):
    """A process manager whom plugins can be registered to, which are then executed regularly."""

    def __init__(self, initial_state=None):
        """Initializes the process manager.

        Arguments:
        initial_state -- An initial god map for this process manager
        """
        self._plugins = OrderedDict()
        self._god_map = GodMap() if initial_state is None else copy(initial_state)
        self.original_universe = initial_state is None


    def register_plugin(self, name, plugin):
        """Registers a plugin with the process manager. The name needs to be unique."""
        if name in self._plugins:
            raise NameConflictException('A plugin with name "{}" already exists.'.format(name))
        self._plugins[name] = plugin

    def start_loop(self):
        """Starts the loop of the project manager.

        Calls start() on all registered plugins. Will continuously update itself, 
        until it is stopped or the ROS node is shut down.
        """
        for plugin in self._plugins.values():
            plugin.start(self._god_map)
        print('init complete')
        while self.update() and not rospy.is_shutdown():
            # TODO make sure this can be properly killed without rospy dependency
            if self.original_universe:
                rospy.sleep(0.1)

    def stop(self):
        """Calls stop() on all registered plugins."""
        for plugin in self._plugins.values():
            plugin.stop()

    def get_god_map(self):
        """Returns the process manager's god map."""
        return self._god_map

    def update(self):
        """Calls update on registered plugins.

        Sequentially calls the update function on registered plugins.
        If a plugin calls for the creation of a new parallel universe,
        it is created, a new process manager is created, replacements for
        the registered plugins are registered to it and the new process manager
        is updated until it terminates. Its resulting god map is copied to the old god map.

        Returns True as long as no plugin calls for the destruction of a parallel universe.
        """
        # TODO doesn't die during planning
        for plugin_name, plugin in self._plugins.items():
            while True:
                plugin.update()
                if plugin.end_parallel_universe():
                    print('destroying parallel universe')
                    return False
                if plugin.create_parallel_universe():
                    print('creating new parallel universe')
                    parallel_universe = ProcessManager(initial_state=self._god_map)
                    for n, p in self._plugins.items():
                        parallel_universe.register_plugin(n, p.get_replacement())
                    t = time()
                    e = None
                    try:
                        parallel_universe.start_loop()
                    except MAX_NWSR_REACHEDException as e:
                        print(e)
                    except Exception as e:
                        traceback.print_exc()
                    finally:
                        print('parallel universe died')
                    parallel_universe.stop()
                    rospy.loginfo('parallel universe existed for {}s'.format(time()-t))

                    # copy new expressions
                    self._god_map.expr_to_key = parallel_universe.get_god_map().expr_to_key
                    self._god_map.key_to_expr = parallel_universe.get_god_map().key_to_expr

                    plugin.post_mortem_analysis(parallel_universe.get_god_map(), e)
                else:
                    break
        return True




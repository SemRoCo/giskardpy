class Plugin(object):

    def start(self):
        raise NotImplementedError('Please implement the start method of this plugin.')

    def stop(self):
        raise NotImplementedError('Please implement the stop method of this plugin.')

class IOPlugin(Plugin):
    def get_readings(self):
        return {}

    def update(self, databus):
        raise NotImplementedError('Please implement the update method of this plugin.')

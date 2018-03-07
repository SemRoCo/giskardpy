class Plugin(object):

    def start(self, databus):
        self.databus = databus

    def stop(self):
        raise NotImplementedError('Please implement the stop method of this plugin.')

class IOPlugin(Plugin):
    def get_readings(self):
        return {}

    def update(self):
        pass

class Plugin(object):
    
    def start(self):
        raise NotImplementedError('Please implement the start method of this plugin.')

    def stop(self):
        raise NotImplementedError('Please implement the stop method of this plugin.')

    def update(self):
        raise NotImplementedError('Please implement the update method of this plugin.')
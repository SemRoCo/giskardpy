import abc


class Logger(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def loginfo(self, msg: str): ...

    @classmethod
    @abc.abstractmethod
    def logwarn(self, msg: str): ...

    @classmethod
    @abc.abstractmethod
    def logerr(self, msg: str): ...

    @classmethod
    @abc.abstractmethod
    def logdebug(self, msg: str): ...

    @classmethod
    @abc.abstractmethod
    def logfatal(self, msg: str): ...

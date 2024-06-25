import abc


class MiddlewareWrapper(abc.ABC):
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

    @classmethod
    @abc.abstractmethod
    def resolve_iri(cls, path: str) -> str: ...


class NoMiddleware(MiddlewareWrapper):

    @classmethod
    def loginfo(self, msg: str):
        print(f'[INFO]: {msg}')

    @classmethod
    def logwarn(self, msg: str):
        print(f'[WARN]: {msg}')

    @classmethod
    def logerr(self, msg: str):
        print(f'[ERROR]: {msg}')

    @classmethod
    def logdebug(self, msg: str):
        print(f'[DEBUG]: {msg}')

    @classmethod
    def logfatal(self, msg: str):
        print(f'[FATAL]: {msg}')

    @classmethod
    def resolve_iri(cls, path: str) -> str:
        return path

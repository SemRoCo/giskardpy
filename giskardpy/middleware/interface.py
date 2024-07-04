import abc


class MiddlewareWrapper(abc.ABC):

    @abc.abstractmethod
    def loginfo(self, msg: str): ...

    @abc.abstractmethod
    def logwarn(self, msg: str): ...

    @abc.abstractmethod
    def logerr(self, msg: str): ...

    @abc.abstractmethod
    def logdebug(self, msg: str): ...

    @abc.abstractmethod
    def logfatal(self, msg: str): ...

    @abc.abstractmethod
    def resolve_iri(cls, path: str) -> str: ...


class NoMiddleware(MiddlewareWrapper):

    def loginfo(self, msg: str):
        print(f'[INFO]: {msg}')

    def logwarn(self, msg: str):
        print(f'[WARN]: {msg}')

    def logerr(self, msg: str):
        print(f'[ERROR]: {msg}')

    def logdebug(self, msg: str):
        print(f'[DEBUG]: {msg}')

    def logfatal(self, msg: str):
        print(f'[FATAL]: {msg}')

    def resolve_iri(cls, path: str) -> str:
        return path

from giskardpy.middleware.interface import MiddlewareWrapper, NoMiddleware

__middleware: MiddlewareWrapper = NoMiddleware()


def set_middleware(middleware: MiddlewareWrapper) -> None:
    global __middleware
    __middleware = middleware


def get_middleware() -> MiddlewareWrapper:
    global __middleware
    return __middleware

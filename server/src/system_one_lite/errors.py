"""Errors that an inference backend may report to the HTTP layer."""


class RequestContractError(ValueError):
    pass


class TooManyOptions(RequestContractError):
    pass

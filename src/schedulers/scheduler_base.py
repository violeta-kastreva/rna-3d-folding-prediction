from typing import Protocol, Self, Any, TypeVar, runtime_checkable

T_co = TypeVar('T_co', covariant=True)


@runtime_checkable
class SchedulerBase(Protocol):
    def step(self, **kwargs) -> None: ...

    @property
    def name(self) -> str: ...

    @property
    def observee(self) -> T_co: ...

    def get_config(self) -> dict[str, Any]: ...

    @classmethod
    def load_from_config(cls, config: dict[str, Any], observee: T_co) -> Self: ...

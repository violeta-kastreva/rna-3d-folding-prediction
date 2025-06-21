from schedulers.scheduler_base import SchedulerBase


class SchedulerManager:
    def __init__(self, scheduler):
        self.schedulers: dict[str, list[tuple[str, SchedulerBase]]] = {}

    def add_pre_epoch(self, scheduler: SchedulerBase) -> SchedulerBase:
        return self._add("pre_epoch", scheduler)

    def add_post_epoch(self, scheduler: SchedulerBase) -> SchedulerBase:
        return self._add("post_epoch", scheduler)

    def add_pre_train_batch(self, scheduler: SchedulerBase) -> SchedulerBase:
        return self._add("pre_train_batch", scheduler)

    def add_post_train_batch(self, scheduler: SchedulerBase) -> SchedulerBase:
        return self._add("post_train_batch", scheduler)

    def add_pre_validation(self, scheduler: SchedulerBase) -> SchedulerBase:
        return self._add("pre_validation", scheduler)

    def add_post_validation(self, scheduler: SchedulerBase) -> SchedulerBase:
        return self._add("post_validation", scheduler)

    def _add(self, category: str, scheduler: SchedulerBase) -> SchedulerBase:
        if category not in self.schedulers:
            self.schedulers[category] = []

        self.schedulers[category].append((scheduler.name, scheduler))

        return scheduler

    def step_pre_epoch(self, **kwargs) -> None:
        self._step("pre_epoch", **kwargs)

    def step_post_epoch(self, **kwargs) -> None:
        self._step("post_epoch", **kwargs)

    def step_pre_train_batch(self, **kwargs) -> None:
        self._step("pre_train_batch", **kwargs)

    def step_post_train_batch(self, **kwargs) -> None:
        self._step("post_train_batch", **kwargs)

    def step_pre_validation(self, **kwargs) -> None:
        self._step("pre_validation", **kwargs)

    def step_post_validation(self, **kwargs) -> None:
        self._step("post_validation", **kwargs)

    def _step(self, category: str, **kwargs) -> None:
        if category not in self.schedulers:
            return

        for _, scheduler in self.schedulers[category]:
            scheduler.step(**kwargs)

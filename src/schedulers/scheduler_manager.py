from schedulers.scheduler_base import SchedulerBase


class SchedulerManager:
    def __init__(self):
        self.schedulers: dict[str, list[tuple[str, SchedulerBase]]] = {}

    def add_on_setup(self, scheduler: SchedulerBase) -> SchedulerBase:
        return self._add("on_setup", scheduler)

    def add_pre_training(self, scheduler: SchedulerBase) -> SchedulerBase:
        return self._add("pre_training", scheduler)

    def add_post_training(self, scheduler: SchedulerBase) -> SchedulerBase:
        return self._add("post_training", scheduler)

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

    def add_pre_validation_batch(self, scheduler: SchedulerBase) -> SchedulerBase:
        return self._add("pre_validation_batch", scheduler)

    def add_post_validation_batch(self, scheduler: SchedulerBase) -> SchedulerBase:
        return self._add("post_validation_batch", scheduler)

    def add_pre_test(self, scheduler: SchedulerBase) -> SchedulerBase:
        return self._add("pre_test", scheduler)

    def add_post_test(self, scheduler: SchedulerBase) -> SchedulerBase:
        return self._add("post_test", scheduler)

    def add_pre_test_batch(self, scheduler: SchedulerBase) -> SchedulerBase:
        return self._add("pre_test_batch", scheduler)

    def add_post_test_batch(self, scheduler: SchedulerBase) -> SchedulerBase:
        return self._add("post_test_batch", scheduler)

    def add_on_train_error(self, scheduler: SchedulerBase) -> SchedulerBase:
        return self._add("on_train_error", scheduler)

    def add_on_test_error(self, scheduler: SchedulerBase) -> SchedulerBase:
        return self._add("on_test_error", scheduler)

    def _add(self, category: str, scheduler: SchedulerBase) -> SchedulerBase:
        if category not in self.schedulers:
            self.schedulers[category] = []

        self.schedulers[category].append((scheduler.name, scheduler))

        return scheduler

    def step_on_setup(self, **kwargs) -> None:
        self._step("on_setup", **kwargs)

    def step_pre_training(self, **kwargs) -> None:
        self._step("pre_training", **kwargs)

    def step_post_training(self, **kwargs) -> None:
        self._step("post_training", **kwargs)

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

    def step_pre_validation_batch(self, **kwargs) -> None:
        self._step("pre_validation_batch", **kwargs)

    def step_post_validation_batch(self, **kwargs) -> None:
        self._step("post_validation_batch", **kwargs)

    def step_pre_test(self, **kwargs) -> None:
        self._step("pre_test", **kwargs)

    def step_post_test(self, **kwargs) -> None:
        self._step("post_test", **kwargs)

    def step_pre_test_batch(self, **kwargs) -> None:
        self._step("pre_test_batch", **kwargs)

    def step_post_test_batch(self, **kwargs) -> None:
        self._step("post_test_batch", **kwargs)

    def step_on_train_error(self, **kwargs) -> None:
        self._step("on_train_error", **kwargs)

    def step_on_test_error(self, **kwargs) -> None:
        self._step("on_test_error", **kwargs)

    def _step(self, category: str, **kwargs) -> None:
        if category not in self.schedulers:
            return

        for _, scheduler in self.schedulers[category]:
            scheduler.step(**kwargs)

    def state_dict(self) -> dict:
        return dict(self.schedulers)

    def load_state_dict(self, state_dict: dict) -> None:
        self.schedulers = state_dict
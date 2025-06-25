from typing import Self

from schedulers.scheduler_manager import SchedulerManager


class SchedulerManagerBuilder:
    def __init__(self):
        self.scheduler_manager: SchedulerManager = SchedulerManager()

    def build_on_setup(self) -> Self:
        return self


    def build(self) -> SchedulerManager:
        return self.scheduler_manager
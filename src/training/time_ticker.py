import time
from typing import Union


class TimeTicker:
    NS_TO_SECONDS_SCALE: float = 1e-9

    def __init__(self):
        self.times: list[tuple[str, int]] = []

    def tick(self, name: str) -> int:
        current_time: int = time.perf_counter_ns()
        self.times.append((name, current_time))
        return current_time

    def get_time(self, name: str) -> int:
        for t_name, t_value in reversed(self.times):
            if t_name == name:
                return t_value

        raise ValueError(f"Time '{name}' not found in ticker.")

    def get_elapsed_secs(self, start_name: str, end_name: str) -> float:
        end_time: int = self.get_time(end_name)
        start_time: int = self._get_time_before(start_name, end_time)

        elapsed_ns: int = end_time - start_time
        return elapsed_ns * self.NS_TO_SECONDS_SCALE

    def get_time_since_start(self, stringify: bool) -> Union[float, str]:
        if not self.times:
            raise ValueError("No times recorded in ticker.")

        start_time: int = self.times[0][1]
        current_time: int = time.perf_counter_ns()
        elapsed_ns: int = current_time - start_time

        if not stringify:
            return elapsed_ns * self.NS_TO_SECONDS_SCALE

        return self.format_nanoseconds(elapsed_ns)

    @staticmethod
    def format_nanoseconds(ns) -> str:
        seconds, nanoseconds = divmod(ns, 1_000_000_000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        seconds += nanoseconds / 1_000_000_000

        return f"{hours} hr {minutes} min {seconds:.2f} sec"

    def _get_time_before(self, name: str, moment: int) -> int:
        for t_name, t_value in reversed(self.times):
            if t_name == name and t_value < moment:
                return t_value

        raise ValueError(f"Time '{name}' not found in ticker.")

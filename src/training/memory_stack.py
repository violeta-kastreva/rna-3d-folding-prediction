from typing import Any


class MemoryStack:
    def __init__(self):
        self.memory: dict[str, Any] = dict()

    def __contains__(self, key: str) -> bool:
        return key in self.memory

    def __getitem__(self, key: str) -> Any:
        if key not in self.memory:
            raise KeyError(f"Key '{key}' not found in memory stack.")
        return self.memory[key]

    def __iter__(self):
        return iter(self.memory)

    def items(self):
        return self.memory.items()

    def to_dict(self) -> dict[str, Any]:
        return dict(self.memory)

    def __setitem__(self, key: str, value: Any) -> None:
        self.memory[key] = value

    def __delitem__(self, key: str) -> None:
        if key not in self.memory:
            raise KeyError(f"Key '{key}' not found in memory stack.")
        del self.memory[key]

    def clear(self) -> None:
        """Clear the memory stack."""
        self.memory.clear()

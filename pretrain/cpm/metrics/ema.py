from typing import Optional


class EMAValue(object):
    def __init__(self, init_value: Optional[float] = None, decay_factor: float = 0.999) -> None:
        super().__init__()
        self._value = init_value
        self._decay_factor = decay_factor

    @property
    def value(self) -> Optional[float]:
        return self._value

    def update(self, value: float) -> None:
        if self._value is None:
            self._value = value
        else:
            self._value = self._decay_factor * self._value + (1 - self._decay_factor) * value

    def update_with_return(self, value: float) -> Optional[float]:
        self.update(value)
        return self._value

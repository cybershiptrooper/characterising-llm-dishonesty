from dataclasses import dataclass


@dataclass
class BaseConfig:
    n: int = 15
    p: float = 0.5
    seed: int = 0
    flip: bool = False

@dataclass
class MultipleConfig(BaseConfig):
    m: int = 2
    maxinum: int = 20


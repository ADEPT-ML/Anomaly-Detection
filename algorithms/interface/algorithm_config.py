import sys
from dataclasses import dataclass, field

__all__ = ["Setting",
           "NumericSetting",
           "IntegerSetting",
           "FloatSetting",
           "SliderSetting",
           "ToggleSetting",
           "Option",
           "OptionSetting",
           "Config"]


@dataclass
class Setting:
    id: str
    name: str
    description: str


@dataclass
class NumericSetting(Setting):
    type: str = field(default="Numeric", init=False)
    default: float
    step: float


@dataclass
class IntegerSetting(NumericSetting):
    step: int = field(default=1, init=False)


@dataclass
class FloatSetting(NumericSetting):
    step: float = field(default=sys.float_info.epsilon, init=False)


@dataclass
class SliderSetting(NumericSetting):
    lowBound: float
    highBound: float
    step: float


@dataclass
class ToggleSetting(Setting):
    type: str = field(default="Toggle", init=False)
    default: bool


@dataclass
class Option():
    name: str
    settings: list[NumericSetting | ToggleSetting]


@dataclass
class OptionSetting(Setting):
    type: str = field(default="Option", init=False)
    default: str
    options: list[Option]


@dataclass
class Config:
    settings: list[Setting] = field(default_factory=lambda: [], init=True)

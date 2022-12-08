import dataclasses
import json
import sys
from dataclasses import dataclass, field


class JSONEncoder(json.JSONEncoder):
    """An enhanced version of the JSONEncoder class containing support for dataclasses."""

    def default(self, o):
        """Adds JSON encoding support for dataclasses.

        This function overrides the default function of JSONEncoder and adds support for encoding dataclasses as JSON.
        Uses the superclass default function for all other types.

        Args:
            o: The object to serialize into a JSON representation.

        Returns:
            The JSON representation of the specified object.
        """

        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


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
    settings: list[Setting]

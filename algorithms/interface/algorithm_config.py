"""Contains all classes to create an algorithm config."""
import dataclasses
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
           "AlgorithmConfig"]


@dataclass(frozen=True)
class Setting:
    """The base settings class."""
    id: str
    name: str
    description: str

    def validate(self) -> None:
        """Validates the settings instance.

        Checks if all attributes are of the correct type.

        Raises:
            TypeError: If an attribute is not of the expected type.
        """
        for attribute in dataclasses.fields(self):
            attr_value = getattr(self, attribute.name)
            if not isinstance(attr_value, attribute.type):
                raise TypeError(f"Expected {attribute.name} to be {attribute.type}, but got {type(attr_value)} instead")

    def __post_init__(self) -> None:
        """Validates the created instance according to its own validation function.

        Raises:
            TypeError: Upon an invalid configuration due to type mismatches.
        """
        self.validate()


@dataclass(frozen=True)
class NumericSetting(Setting):
    """The settings class for all numeric settings."""
    type: str = field(default="Numeric", init=False)
    default: float
    step: float

    def validate(self) -> None:
        """Validates the numeric settings instance.

        Checks if the default and step values are valid.

        Raises:
            ValueError: If the step or default value is invalid.
        """
        if self.step <= 0:
            raise ValueError(f"Step is not allowed to be equal or less than zero but is {self.step}")
        if self.default % self.step:
            raise ValueError(f"The default value is not dividable by the specified step value.")

    def __post_init__(self) -> None:
        """Validates the created instance according to its own and its superclass validation function.

        Raises:
            TypeError: Upon an invalid configuration due to type mismatches.
            ValueError: Upon invalid variable values.
        """
        super().__post_init__()
        self.validate()


@dataclass(frozen=True)
class IntegerSetting(NumericSetting):
    """The settings class for all settings with integer values."""
    step: int = field(default=1, init=False)


@dataclass(frozen=True)
class FloatSetting(NumericSetting):
    """The settings class for all settings with float values."""
    step: float = field(default=sys.float_info.epsilon, init=False)


@dataclass(frozen=True)
class SliderSetting(NumericSetting):
    """The settings class for all settings that can be displayed as a slider."""
    lowBound: float
    highBound: float
    step: float

    def validate(self) -> None:
        """Validates the slider settings instance.

        Checks if the bounds and default value are valid.

        Raises:
            ValueError: If the bounds or default value is invalid.
        """
        if self.lowBound >= self.highBound:
            raise ValueError("The bounds do not describe a valid range.")
        if not self.lowBound <= self.default <= self.highBound:
            raise ValueError("The default value is not within the specified range.")

    def __post_init__(self) -> None:
        """Validates the created instance according to its own and its superclass validation function.

        Raises:
            TypeError: Upon an invalid configuration due to type mismatches.
            ValueError: Upon invalid variable values.
        """
        super().__post_init__()
        self.validate()


@dataclass(frozen=True)
class ToggleSetting(Setting):
    """The settings class for all settings with boolean values."""
    type: str = field(default="Toggle", init=False)
    default: bool


@dataclass(frozen=True)
class Option:
    """Class for an option in a dropdown menu."""
    name: str
    settings: list[NumericSetting | ToggleSetting] = field(default_factory=list)

    def validate(self) -> None:
        """Validates the option instance.

        Checks if the settings list is valid.

        Raises:
            TypeError: If the settings list is invalid.
        """
        if not all(isinstance(e, NumericSetting | ToggleSetting) for e in self.settings):
            raise TypeError("Settings is expected to only contain a list of NumericSetting or ToggleSetting objects.")

    def __post_init__(self) -> None:
        """Validates the created instance according to its own validation function.

        Raises:
            TypeError: Upon an invalid configuration due to type mismatches.
        """
        self.validate()


@dataclass(frozen=True)
class OptionSetting(Setting):
    """Class for a dropdown menu."""
    type: str = field(default="Option", init=False)
    default: str
    options: list[Option]

    def validate(self) -> None:
        """Validates the option instance.

        Checks if the settings list is valid.

        Raises:
            TypeError: If the options list contains an invalid type.
            ValueError: If any of the variables contain invalid data.
        """
        option_names = [o.name for o in self.options]
        if not all(isinstance(e, Option) for e in self.options):
            raise TypeError("Options is expected to only contain a list of Option objects.")
        if len(self.options) < 2:
            raise ValueError("At least two options are required.")
        if self.default not in (e.name for e in self.options):
            raise ValueError("The specified default is not in the list of options.")
        if len(option_names) != len(set(option_names)):
            raise ValueError("Duplicate option names")

    def __post_init__(self) -> None:
        """Validates the created instance according to its own and its superclass validation function.

        Raises:
            TypeError: Upon an invalid configuration due to type mismatches.
            ValueError: Upon invalid variable values.
        """
        super().__post_init__()
        self.validate()


@dataclass(frozen=True)
class AlgorithmConfig:
    """Class that contains all information regarding the config options of an algorithm."""
    settings: list[Setting] = field(default_factory=list, init=True)

    def validate(self) -> None:
        """Validates the option instance.

        Checks if the settings list is valid.

        Raises:
            TypeError: If the options list is not a list or contains an invalid type.
            ValueError: If multiple settings share the same ID.
        """
        setting_ids = []
        for s in self.settings:
            setting_ids.append(s.id)
            if isinstance(s, OptionSetting):
                setting_ids.extend(s.id for o in s.options for s in o.settings)

        if not isinstance(self.settings, list):
            raise TypeError("Settings is expected to be a list of Setting objects.")
        if not all(isinstance(e, Setting) for e in self.settings):
            raise TypeError("The settings list is expected to only contain a list of Setting objects.")
        if len(setting_ids) != len(set(setting_ids)):
            raise ValueError("Duplicate setting IDs.")

    def __post_init__(self) -> None:
        """Validates the created instance according to its own validation function.

        Raises:
            TypeError: Upon an invalid configuration due to type mismatches.
            ValueError: Upon invalid variable values.
        """
        self.validate()

from dataclasses import dataclass, field
from typing import Dict, Pattern, Iterable, NamedTuple
from types import MappingProxyType
import re
from enum import Enum
from pathlib import Path
from decimal import Decimal
import json


class SweepDirection(Enum):
    INCREASING = 1
    DECREASING = -1
    UNDEFINED = 0

    __aliases_increasing = frozenset(('inc', 'up', 'increasing'))
    __aliases_decreasing = frozenset(('dec', 'down', 'decreasing'))
    __aliases_undefined = frozenset(('und', 'na', 'undefined'))

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.strip().lower()
            if value in cls.__aliases_increasing:
                return cls.INCREASING
            elif value in cls.__aliases_decreasing:
                return cls.DECREASING
            elif value in cls.__aliases_undefined:
                return cls.UNDEFINED
    
    def __eq__(self, other):
        if isinstance(other, str):
            other = self._missing_(other)
        return super().__eq__(other)
    
    def __hash__(self):
        return super().__hash__()
    
@dataclass
class ParameterDefinition:
    """Immutable parameter definition that can be shared"""
    name_id: str
    long_names: str| Iterable[str]
    short_names: str | Iterable[str] = frozenset()
    units: str | Iterable[str] = frozenset()
    special_values: MappingProxyType[str, Decimal] = field(
        default_factory=lambda: MappingProxyType({}))
    patterns: MappingProxyType[str, Pattern] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False)

    def __post_init__(self):
        self._to_immutables()
        self._validate_names()
        self._compile_patterns()

    def _to_immutables(self):
        """Convert to immutable types"""
        self.long_names = frozenset(self.long_names)
        self.short_names = frozenset(self.short_names)
        self.units = frozenset(self.units)
        self.special_values = MappingProxyType(
            {key: Decimal(str(val)) for key, val in (self.special_values or {}).items()})
        if not isinstance(self.patterns, MappingProxyType):
            self.patterns = MappingProxyType(self.patterns or {})
    
    def _validate_names(self) -> None:
        if not (self.long_names or self.short_names):
            raise ValueError("Long and short names cannot be empty simultaneously")
        overlap = self.long_names & self.short_names
        if overlap:
            raise ValueError(f"Long and short names overlap: {overlap}")
        
    def _create_base_patterns(self) -> Dict[str, str]:
        special_values_pattern = '|'.join(map(re.escape, self.special_values.keys()))
        return {
            'LNAME': '|'.join(map(re.escape, self.long_names)),
            'SNAME': '|'.join(map(re.escape, self.short_names)),
            'NAME': '|'.join(map(re.escape, self.long_names | self.short_names)),
            'UNIT': '|'.join(map(re.escape, self.units)),
            'VALUE': rf'-?\d+\.?\d*(?:[eE][+-]?\d+)?|{special_values_pattern}',
        }

    def _compile_patterns(self) -> None:
        base = self._create_base_patterns()
        
        # Default patterns
        default_patterns = {
            'sweep': r'sweep{NAME}|{NAME}sweep',
            'range': r'{NAME}{VALUE}to{VALUE}{UNIT}?',
            'fixed': r'{NAME}=?{VALUE}{UNIT}?'
        }

        # Merge with custom patterns if provided
        patterns_to_compile = default_patterns
        if self.patterns:
            patterns_to_compile.update(self.patterns)

        # Replace placeholders and compile
        compiled_patterns = dict()
        for pattern_name, pattern_template in patterns_to_compile.items():
            # Replace all placeholders with actual patterns
            final_pattern = pattern_template
            for placeholder, value in base.items():
                if placeholder == 'VALUE' or placeholder == 'UNIT':
                    final_pattern = final_pattern.replace(
                        f'{{{placeholder}}}',
                        f'({value})')
                else:
                    final_pattern = final_pattern.replace(
                        f'{{{placeholder}}}',
                        f'(?:{value})')

            try:
                compiled_patterns[pattern_name] = re.compile(final_pattern)
            except re.error as e:
                raise ValueError(f"Invalid pattern '{pattern_name}': {e}")
        self.patterns = MappingProxyType(compiled_patterns)

    def get_pattern(self, pattern_type: str) -> Pattern:
        if pattern_type not in self.patterns:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        return self.patterns[pattern_type]

    def match(self, pattern_type: str, text: str) -> re.Match | None:
        return self.get_pattern(pattern_type).match(text)

    def search(self, pattern_type: str, text: str) -> re.Match | None:
        return self.get_pattern(pattern_type).search(text)
    
@dataclass
class _ParameterState:
    """Mutable instance-specific state"""
    value: Decimal | None = None
    is_swept: bool = False
    min_val: Decimal | None = None
    max_val: Decimal | None = None
    sweep_direction: SweepDirection = SweepDirection.UNDEFINED
    unit: str | None = None
    
    def __eq__(self, other):
        for attr, value in self.__dict__.items():
            if value != other.__dict__.get(attr):
                return False
        return True
    
    def __copy__(self):
        # Method is not tested
        return type(self)(**self.__dict__.copy())
    
class Parameter:
    def __init__(self, definition: ParameterDefinition, **kwargs):
        self.definition = definition
        self.state = _ParameterState(**kwargs)
            
    def parse_fixed(self, text: str) -> bool:
        """Parse fixed parameter value from regex match"""
        m = self.definition.match('fixed', text)
        if not m:
            return False
        value_str = m.group(1)
        unit_str = m.group(2)
        self.set_fixed(self._value2decimal(value_str))
        self.state.unit = unit_str
        return True
    
    def parse_range(self, text: str) -> bool:
        m = self.definition.search('range', text)
        if not m:
            return False
        start_str, end_str = m.group(1), m.group(2)
        unit_str = m.group(3)
        self.set_swept(
            self._value2decimal(start_str),
            self._value2decimal(end_str)
            )
        self.state.unit = unit_str
        return True

    def parse_sweep(self, text: str) -> bool:
        is_range = self.parse_range(text)
        if is_range:
            return True
        
        m = self.definition.match('sweep', text)
        if not m:
            return False
        self.state.is_swept =True
        return True
    
    def set_fixed(self, value: Decimal):
        self.state.is_swept = False
        self.state.value = value
        return self
    
    def set_swept(self, start_val: Decimal, end_val: Decimal):
        self.state.is_swept = True
        if start_val is not None and end_val is not None:
            if end_val > start_val:
                self.state.sweep_direction = SweepDirection.INCREASING
            elif end_val < start_val:
                start_val, end_val = end_val, start_val
                self.state.sweep_direction = SweepDirection.DECREASING
            self.state.min_val = start_val
            self.state.max_val = end_val
        return self
    
    def update(self, other):
        self.state.value = other.state.value or self.state.value
        self.state.is_swept = other.state.is_swept or self.state.is_swept
        self.state.min_val = other.state.min_val or self.state.min_val
        self.state.max_val = other.state.max_val or self.state.max_val
        self.state.sweep_direction = (other.state.sweep_direction or 
                                      self.state.sweep_direction)
        self.state.unit = other.state.unit or self.state.unit
    
    def _value2decimal(self, value_str):
        """Convert string value to Decimal, handling special values"""
        if value_str in self.definition.special_values:
            return self.definition.special_values[value_str]
        return Decimal(value_str)
    
    def copy(self):
        return self.__copy__()
    
    def __copy__(self):
        return type(self)(self.definition,
                          **self.state.__dict__.copy())
    
    def __str__(self):
        if self.definition.long_names:
            lname = sorted(list(self.definition.long_names))[0]
        else:
            lname = None
        if self.definition.short_names:
            sname = sorted(list(self.definition.short_names))[0]
        else:
            sname = None

        if self.state.is_swept:
            name = lname or sname
            s = f'sweep{name}'
            if self.state.min_val and self.state.max_val:
                mn, mx = self.state.min_val, self.state.max_val
                if self.state.sweep_direction == SweepDirection.DECREASING:
                    mx, mn = mn, mx
                s += f'{mn}to{mx}{self.state.unit or ""}'
            return s
        else:
            name = sname or lname
            return f'{name}={self.state.value}{self.state.unit or ""}'
        
    
    def __repr__(self):
        return ('Parameter: '
                f'[type: {self.definition.name_id}, '
                f'value: {self.state.value}, is_swept: {self.state.is_swept}, '
                f'min_val: {self.state.min_val}, max_val: {self.state.max_val}, '
                f'sweep_direction: {self.state.sweep_direction}]')

class DefinitionsLoader:
    def __init__(self, custom_config_path: Path | str = None):
        self._definitions = {}
        self._load_defaults()
        if custom_config_path:
            self.load_definitions(custom_config_path)

    def _load_defaults(self):
        """Load built-in parameter definitions"""
        default_path = Path(__file__).parent / 'parameter_definitions.json'
        self.load_definitions(default_path)
            
    def load_definitions(self, path: Path | str):
        """Load custom parameter definitions from YAML"""
        with open(path, 'r') as f:
            definitions = json.load(f)

        for name, definition in definitions.items():
            # Convert lists to frozensets and MappingProxyType
            definition['name_id'] = name
            definition['long_names'] = frozenset(definition.get('long_names', []))
            definition['short_names'] = frozenset(definition.get('short_names', []))
            definition['units'] = frozenset(definition.get('units', []))
            definition['special_values'] = MappingProxyType(definition.get('special_values', {}))
            definition['patterns'] = MappingProxyType(definition.get('patterns', {}))
            self._definitions[name] = definition

    def get_definition_dict(self, name: str) -> dict[str, set]:
        """Get parameter definition by name"""
        if name not in self._definitions:
            raise KeyError(f"Parameter definition '{name}' not found")
        return self._definitions[name]
    
    def get_definition(self, name: str) -> Parameter:
        """Create Parameter instance from definition"""
        definition = self.get_definition_dict(name)
        return ParameterDefinition(**definition)
    
    def get_all(self) -> list[Parameter]:
        return [self.get_definition(name) for name in self._definitions.keys()]

class ParameterState(NamedTuple):
    value: float | None
    is_swept: bool
    min: float | None
    max: float | None
    sweep_direction: SweepDirection
    unit: str | None

    @staticmethod
    def to_float(val) -> float | None:
        return float(val) if val is not None else None

    @classmethod
    def from_state(cls, state: _ParameterState):
        return cls(
            value=cls.to_float(state.value),
            is_swept=state.is_swept,
            min=cls.to_float(state.min_val),
            max=cls.to_float(state.max_val),
            sweep_direction=state.sweep_direction,
            unit=state.unit
            )
    
    @property
    def range(self) -> tuple[float, float] | None:
        if self.is_swept:
            return (self.min, self.max)
        return None
    
    @property
    def sweep(self) -> tuple[float, float, SweepDirection] | None:
        if self.is_swept:
            return (self.min, self.max, self.sweep_direction)
        return None
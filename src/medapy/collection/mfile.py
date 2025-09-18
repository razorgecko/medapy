from dataclasses import dataclass
import re
from enum import Enum
from pathlib import Path
from decimal import Decimal
from typing import Iterable

from .parameter import (ParameterDefinition,
                        DefinitionsLoader,
                        Parameter,
                        ParameterState)
from medapy.utils import validations

class PolarizationType(Enum):
    CURRENT = 'I'
    VOLTAGE = 'V'

    __aliases_current = frozenset(('i', 'current'))
    __aliases_voltage = frozenset(('v', 'voltage'))

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.strip().lower()
            if value in cls.__aliases_current:
                return cls.CURRENT
            elif value in cls.__aliases_voltage:
                return cls.VOLTAGE
    
    def __eq__(self, other):
        if isinstance(other, str):
            other = self._missing_(other)
        return super().__eq__(other)
    
    def __hash__(self):
        return super().__hash__()

contact_pattern = re.compile(r'([IV])(\d+)(?:-(\d+))?(?:\((-?\d+\.?\d*(?:[eE][+-]?\d+)?[fpnumkMGT]?[AV])\))?')

@dataclass
class ContactPair:
    # For single contact, second_contact will be None
    first_contact: int | None = None
    second_contact: int | None = None
    polarization: PolarizationType | None = None
    magnitude: Decimal | None = None

    def __copy__(self):
        return type(self)(first_contact=self.first_contact,
                          second_contact=self.second_contact,
                          polarization=self.polarization,
                          magnitude=self.magnitude)
        
    def __str__(self) -> str:
        result = f"{self.first_contact}"
        if self.second_contact is not None:
            result += f"-{self.second_contact}"

        if self.polarization is not None:
            result = f"{self.polarization.value}" + result

            if self.magnitude is not None:
                if self.polarization == PolarizationType.CURRENT:
                    unit = 'A'
                else:
                    unit = 'V'
                result += "({magn:{fmt}}{unit})".format(magn=self.magnitude,
                                                        fmt='.2g' if 0.01 <= self.magnitude <= 100 else '.1e',
                                                        unit=unit)
        return result
    
    def __post_init__(self):
        if isinstance(self.polarization, str):
            self.polarization = PolarizationType(self.polarization)
        if isinstance(self.magnitude, (str, int, float)):
            self.magnitude = Decimal(str(self.magnitude))
    
    def parse_contacts(self, text: str) -> bool:
        m = contact_pattern.match(text)
        if not m:
            return False
        type_str, first, second, magnitude = m.groups()
        self.first_contact = int(first)
        self.second_contact = int(second) if second else None
        self.polarization = PolarizationType(type_str)
        self.magnitude = self._convert_magntude(magnitude) if magnitude else None
        return True
    
    def pair_matches(self, pair: Iterable[int] | int | 'ContactPair') -> bool:
        if isinstance(pair, int):
            return (self.first_contact == pair and 
                    self.second_contact is None)
        elif isinstance(pair, ContactPair):
            return self == pair
        return (self.first_contact == pair[0] and 
                self.second_contact == pair[1])
    
    def to_tuple(self):
        return (self.first_contact, self.second_contact,
                self.polarization, self.magnitude)
    
    def copy(self):
        return self.__copy__()
    
    def polarized(self, polarization, magnitude=None):
        return type(self)(self.first_contact,
                          self.second_contact,
                          polarization=polarization,
                          magnitude=magnitude)

    def _convert_magntude(self, magnitude):
        return Decimal(magnitude.replace('f', 'e-15')
                       .replace('p', 'e-12')
                       .replace('n', 'e-9')
                       .replace('u', 'e-6')
                       .replace('m', 'e-3')
                       .replace('k', 'e3')
                       .replace('M', 'e6')
                       .replace('G', 'e9')
                       .replace('T', 'e12')
                       .rstrip('AV'))
    
    def __hash__(self):
        return hash((self.first_contact, self.second_contact, self.polarization, self.magnitude))

@dataclass(frozen=False)
class MeasurementFile:
    path: Path
    parameters: dict[str, Parameter]
    contact_pairs: list[ContactPair]
    separator: str = "_"

    def __init__(self, 
                 path: str | Path,
                 parameters: list[ParameterDefinition | Parameter] | Path | str,
                 separator: str = "_"):
        """
        Initialize MeasurementFile

        Args:
            path: Path to the measurement file
            parameters: Either a list of Parameter instances or path to parameter definitions file
            separator: Filename parts separator
        """
        self.path = Path(path)
        self.separator = separator
        self.contact_pairs = []

        # Initialize parameters dictionary
        if isinstance(parameters, (str, Path)):
            param_defs = DefinitionsLoader(parameters)
            self.param_definitions = {dfn.name_id: dfn for dfn in param_defs.get_all()}
        else:
            item_type = type(parameters[0])
            validations.class_in_iterable(parameters, item_type, iter_name='parameters')
            # Convert list of parameters to dictionary
            self.param_definitions = dict()
            if item_type == Parameter:
                self.parameters = dict()
                for param in parameters:
                    name = param.definition.name_id
                    self.param_definitions[name] = param.definition
                    self.parameters[name] = param.copy()
                return
            if item_type == ParameterDefinition:
                for param in parameters:
                    name = param.name_id
                    self.param_definitions[name] = param
                    
        self.parameters = dict()
        self._parse_filename()

    @property
    def name(self) -> str:
        return self.path.name
    
    def check(self,
              contacts: tuple[int, int] | list[tuple[int, int] | int] | int | None = None,
              polarization: str | None = None,
              sweep_direction: str | None = None,
              **parameter_filters: dict) -> bool:
        """Check if file matches all filter conditions"""
                
        # Check contacts
        if contacts is not None:
            if not self.check_contacts(contacts):
                return False

        # Check polarization
        if polarization is not None:
            if not self.check_polarization(polarization):
                return False

        # Check sweep direction
        if sweep_direction is not None:
            if not self.check_sweep_direction(sweep_direction):
                return False

        # Check parameter filters
        for param_name, filter_value in parameter_filters.items():
            if not self.check_parameter(param_name, filter_value):
                return False
            
        return True

    def check_polarization(self, polarization: str):
        return any(pair.type == polarization for pair in self.contact_pairs)
    
    def check_sweep_direction(self, sweep_direction: str):
        return any(param.state.sweep_direction == sweep_direction 
                    for param in self.parameters.values())
            
    def check_contacts(self,
                       contacts: tuple[int, int] | list[tuple[int, int] | int] | int) -> bool:
        """Check if file contains specified contact configuration"""

        # Convert single pair/contact to list
        if not isinstance(contacts, list):
            contacts = [contacts]

        # Check if all specified contacts/pairs are present
        return all(
            any(pair.pair_matches(check_pair) 
                for pair in self.contact_pairs)
            for check_pair in contacts
        )

    def check_parameter(self,
                        name: str,
                        value) -> bool:
        """Check if parameter matches value or range"""
        param = self.parameters.get(name)
        if not param:
            return False

        # Handle exact value
        if not isinstance(value, Iterable):
            if param.state.is_swept:
                return False
            return param.state.value == param._value2decimal(str(value))

        # Handle range
        try:
            min_val, max_val = map(param._value2decimal, value)
            if min_val > max_val:
                min_val, max_val = max_val, min_val
        except ValueError:
            raise ValueError("Param range length should be 2; "
                             f"got {len(value)} for {name}")
        if param.state.is_swept:
            # For swept parameter, check if sweep range overlaps with filter range
            return (param.state.min_val <= max_val and 
                   param.state.max_val >= min_val)

        # For fixed parameter, check if value is within range
        return min_val <= param.state.value <= max_val
    
    def get_parameter(self, name: str) -> Parameter:
        param = self.parameters.get(name)
        if not param:
            raise ValueError(f'{name} parameter is not defined for file {self.path.name}')
        return param
    
    def state_of(self, name: str) -> ParameterState:
        param = self.get_parameter(name)
        return ParameterState.from_state(param.state)
    
    def _parse_filename(self) -> None:
        # Get filename without extension
        name = self.path.stem
        # Split by separator
        name_parts = name.split(self.separator)

        for part in name_parts:
            self._parse_part(part)

    def _parse_part(self, part: str) -> None:
        # Try to parse as contact pair first
        contact_pair = ContactPair()
        is_contact = contact_pair.parse_contacts(part)
        if is_contact:
            self.contact_pairs.append(contact_pair)
            return

        # Try to parse as parameter
        for param_def in self.param_definitions.values():
            # Try as sweep
            param_name = param_def.name_id
            param = Parameter(param_def)
            is_sweep = param.parse_sweep(part)
            if is_sweep:
                try:
                    self.parameters[param_name].update(param)
                except KeyError:
                    self.parameters[param_name] = param
                continue
            is_fixed = param.parse_fixed(part)
            if is_fixed:
                try:
                    self.parameters[param_name].update(param)
                except KeyError:
                    self.parameters[param_name] = param
                continue
    
    def merge(self,
              other: 'MeasurementFile',
              strict_mode: bool = False,
              ) -> 'MeasurementFile':
        """
        Merge this file representation with another one.

        Args:
            other: Another FileRepresentation to merge with
            strict_mode: If True, verify all parameters are equal before merging

        Returns:
            A new MeasurementFile with merged parameters and contact pairs

        Raises:
            ValueError: If strict_mode is True and parameters differ between files
        """
        # Check parameters in strict mode
        if strict_mode:
            for param_name, param in self.parameters.items():
                if param_name in other.parameters:
                    other_param = other.parameters[param_name]
                    # Check if parameters are equal
                    if (param.state != other_param.state):
                        raise ValueError(f"Parameter '{param_name}' differs between files in strict mode")
                
        # Merge parameters (self take precedence in case of conflict)
        merged_parameters = {}
        merged_parameters.update(other.parameters)
        merged_parameters.update(self.parameters)
        parameters_list = [param for param in merged_parameters.values()]
        
        # Merge contact pairs (removing duplicates)
        merged_contacts = []
        seen_contacts = set()

        # Add contacts from self
        for contact in self.contact_pairs:
            key = contact.to_tuple()
            if key not in seen_contacts:
                merged_contacts.append(contact.copy())
                seen_contacts.add(key)

        # Add contacts from other
        for contact in other.contact_pairs:
            key = contact.to_tuple()
            if key not in seen_contacts:
                merged_contacts.append(contact.copy())
                seen_contacts.add(key)

        # Create a new MeasurementFile with merged data
        # Use the separator from the current instance
        merged_file = type(self)(
            path=self.path,
            parameters=parameters_list,
            separator=self.separator
        )
        merged_file.contact_pairs = merged_contacts
        
        merged_filename = merged_file.generate_filename()
        merged_file.path = merged_file.path.parent / merged_filename
        return merged_file
    
    def rename(self,
               directory: str | Path | None = None,
               name: str | Path | None = None,
               prefix: str | None = None,
               postfix: str | None = None,
               sep: str | None = None,
               ext: str | None = None
               ) -> None:
        # Change separator
        if sep:
            self.separator = sep
        
        # Generate new path
        self.path = self._generate_path(directory=directory, name=name,
                                        prefix=prefix, postfix=postfix,
                                        sep=sep, ext=ext)
    
    def _generate_path(self,
                       directory: str | Path | None = None,
                       name: str | Path | None = None,
                       prefix: str | None = None,
                       postfix: str | None = None,
                       sep: str | None = None,
                       ext: str | None = None
                       ) -> Path:
        if not directory:
            directory = self.path.parent
        directory = Path(directory).expanduser()
        
        if not name:
            name = self.path.stem
        
        if sep:
            name = name.replace(self.separator, sep)
            self.separator = sep
            
        if prefix:
            name = self.separator.join([prefix, name])
            
        if postfix:
            name = self.separator.join([name, postfix])
            
        if ext and not ext.startswith('.'):
            ext = f".{ext}"
        elif ext is None:
            ext = self.path.suffix
        else:
            ext = ''
        
        return directory / f"{name}{ext}"
    
    def copy(self):
        return self.__copy__()
        
    def __copy__(self):
        new = type(self)(
            path=self.path,
            parameters=[param for param in self.parameters.values()],
            separator=self.separator)
        new.contact_pairs = [pair.copy() for pair in self.contact_pairs]
        return new
            
    def generate_filename(self,
                          prefix: str = None,
                          postfix: str = None,
                          sep: str = None,
                          ext: str = None
                          ) -> str:
        """
        Generate a filename based on stored parameters and contact pairs.

        Args:
            prefix: Optional prefix for the filename
            postfix: Optional postfix for the filename
            sep: Optional separator (instance separator if None)
            ext: File extension (instance extension if None)

        Returns:
            A string representing the new filename
        """
        # Use instance separator if not provided
        sep = sep if sep is not None else self.separator

        # Build the contact part of the filename
        contact_parts = []
        for contacts in self.contact_pairs:
                contact_parts.append(str(contacts))

        # Determine parameters order
        parameters_ordered = []
        # Add sweeping parameters
        for param in self.parameters.values():
            if param.state.is_swept:
                parameters_ordered.append(param)
        # Add fixed parameters
        for param in self.parameters.values():
            if not param.state.is_swept:
                parameters_ordered.append(param)
        
        param_parts = []
        # Build the parameter part of the filename       
        for param in parameters_ordered:
            param_parts.append(str(param))

        # Combine all parts
        filename_parts = []
        if prefix:
            filename_parts.append(prefix)
        filename_parts.extend(contact_parts)
        filename_parts.extend(param_parts)
        if postfix:
            filename_parts.append(postfix)

        # Join with separator and add extension
        filename = sep.join(filename_parts)

        # Add extension if it doesn't already have one
        if ext and not ext.startswith('.'):
            ext = f".{ext}"
        elif ext is None:
            ext = self.path.suffix
        else:
            ext = ''

        return f"{filename}{ext}"

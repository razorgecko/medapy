from dataclasses import dataclass
import re
from enum import Enum
from pathlib import Path
from decimal import Decimal
from typing import Iterable

from .parameter import ParameterDefinition, DefinitionsLoader, Parameter


class Polarization(Enum):
    CURRENT = "I"
    VOLTAGE = "V"

contact_pattern = re.compile(r'([IV])(\d+)(?:-(\d+))?(?:\((-?\d+\.?\d*(?:[eE][+-]?\d+)?[fpnumkMGT]?[AV])\))?')

@dataclass
class ContactPair:
    # For single contact, second_contact will be None
    first_contact: int | None = None
    second_contact: int | None = None
    type: Polarization | None = None
    magnitude: Decimal | None = None

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = Polarization(self.type)
        if isinstance(self.magnitude, (str, int, float)):
            self.magnitude = Decimal(str(self.magnitude))
    
    def parse_contacts(self, text: str) -> bool:
        m = contact_pattern.match(text)
        if not m:
            return False
        type_str, first, second, magnitude = m.groups()
        self.first_contact = int(first)
        self.second_contact = int(second) if second else None
        self.type = Polarization(type_str)
        self.magnitude = self._convert_magntude(magnitude) if magnitude else None
        return True
    
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
    
    def pair_matches(self, pair: Iterable[int] | int | 'ContactPair') -> bool:
        if isinstance(pair, int):
            return (self.first_contact == pair and 
                    self.second_contact is None)
        elif isinstance(pair, ContactPair):
            return self == pair
        return (self.first_contact == pair[0] and 
                self.second_contact == pair[1])
        
    def __str__(self) -> str:
        contacts = f"{self.first_contact}"
        if self.second_contact is not None:
            contacts += f"-{self.second_contact}"

        result = f"{self.type.value}{contacts}"
        if self.magnitude is not None:
            result += f"({self.magnitude})"
        return result

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
            # Convert list of parameters to dictionary
            self.param_definitions = {}
            for param in parameters:
                name = param.name_id
                self.param_definitions[name] = param
                
        self.parameters = dict()
        self._parse_filename()

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
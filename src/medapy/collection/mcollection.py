from typing import Iterator, Union, Tuple, Iterable
from pathlib import Path
from decimal import Decimal

from medapy.collection import (MeasurementFile,
                               Polarization,
                               SweepDirection,
                               ContactPair,
                               ParameterDefinition)
                               

class MeasurementCollection:
    def __init__(self, 
                 collection: str | Path | Iterable[MeasurementFile],
                 parameters: Iterable[ParameterDefinition],
                 file_pattern: str = "*.*",
                 separator: str = "_"):
        
        
        self.param_definitions = {param.name_id: param for param in parameters}
        self.separator = separator

        if isinstance(collection, (list, tuple)):
            try:
                self.folder_path = collection[0].path.parent
                self.files = collection.copy()
            except IndexError:
                self.folder_path = ''
                self.files = []
            except KeyError:
                raise ValueError("collection iterable does not contain measurement files")
            
        else:  
            self.folder_path = Path(collection)
            self.files = []
            for f in self.folder_path.glob(file_pattern):
                if f.is_file():
                    self.files.append(MeasurementFile(f, parameters, separator))
            
    def filter_generator(self, 
               contacts: Union[Tuple[int, int], list[Union[Tuple[int, int], int]], int] = None,
               polarization: Union[str, Polarization] = None,
               sweep_direction: Union[str, SweepDirection] = None,
               **parameter_filters) -> Iterator[MeasurementFile]:
        """
        Filter measurement files based on various criteria, returning a generator

        Args:
            contacts: Single contact pair (1, 2), list of pairs/contacts [(1, 2), 3], or single contact
            polarization: 'I' for current or 'V' for voltage
            sweep_direction: 'up' or 'down'
            **parameter_filters: Parameter name with value or (min, max) tuple

        Returns:
            Iterator of matching MeasurementFile instances
        """
        for meas_file in self.files:
            if self._matches_all_filters(
                meas_file, 
                contacts, 
                polarization, 
                sweep_direction, 
                parameter_filters
            ):
                yield meas_file

    def filter(self,
               contacts: Union[Tuple[int, int], list[Union[Tuple[int, int], int]], int] = None,
               polarization: Union[str, Polarization] = None,
               sweep_direction: Union[str, SweepDirection] = None,
               **parameter_filters) -> 'MeasurementCollection':
        """
        Filter measurement files based on various criteria, returning a new collection

        Args:
            contacts: Single contact pair (1, 2), list of pairs/contacts [(1, 2), 3], or single contact
            polarization: 'I' for current or 'V' for voltage
            sweep_direction: 'up' or 'down'
            **parameter_filters: Parameter name with value or (min, max) tuple

        Returns:
            New MeasurementCollection containing only matching files
        """
        filtered_files = list(self.filter_generator(
            contacts,
            polarization,
            sweep_direction,
            **parameter_filters
        ))
        return MeasurementCollection(
            filtered_files,
            parameters=self.param_definitions.values(),
            separator=self.separator
        )
    
    def _matches_all_filters(self,
                           meas_file: MeasurementFile,
                           contacts,
                           polarization,
                           sweep_direction,
                           parameter_filters: dict) -> bool:
        """Check if file matches all filter conditions"""
                
        # Check contacts
        if contacts is not None:
            if not self._check_contacts(meas_file.contact_pairs, contacts):
                return False

        # Check polarization
        if polarization is not None:
            pol = (polarization if isinstance(polarization, Polarization) 
                   else Polarization(polarization))
            if not any(pair.type == pol for pair in meas_file.contact_pairs):
                return False

        # Check sweep direction
        if sweep_direction is not None:
            direction = (sweep_direction if isinstance(sweep_direction, SweepDirection)
                       else SweepDirection(sweep_direction.lower()))
            if not any(param.state.sweep_direction == direction 
                      for param in meas_file.parameters.values()):
                return False

        # Check parameter filters
        for param_name, filter_value in parameter_filters.items():
            if not self._check_parameter(meas_file, param_name, filter_value):
                return False
            
        return True

    def _check_contacts(self,
                       file_pairs: list[ContactPair],
                       filter_contacts) -> bool:
        """Check if file contains specified contact configuration"""

        # Convert single pair/contact to list
        if not isinstance(filter_contacts, list):
            filter_contacts = [filter_contacts]

        # Check if all specified contacts/pairs are present
        return all(
            any(pair.pair_matches(filter_pair) 
                for pair in file_pairs)
            for filter_pair in filter_contacts
        )

    def _check_parameter(self,
                        meas_file: MeasurementFile,
                        param_name: str,
                        filter_value) -> bool:
        """Check if parameter matches filter value or range"""
        param = meas_file.parameters.get(param_name)
        if not param:
            return False

        # Handle exact value
        if not isinstance(filter_value, Iterable):
            if param.state.is_swept:
                return False
            return param.state.value == Decimal(str(filter_value))

        # Handle range
        try:
            min_val, max_val = map(lambda x: Decimal(str(x)), filter_value)
            if min_val > max_val:
                min_val, max_val = max_val, min_val
        except ValueError:
            raise ValueError("Param range length should be 2; "
                             f"got {len(filter_value)} for {param_name}")
        if param.state.is_swept:
            # For swept parameter, check if sweep range overlaps with filter range
            return (param.state.min_val <= max_val and 
                   param.state.max_val >= min_val)

        # For fixed parameter, check if value is within range
        return min_val <= param.state.value <= max_val

    def __iter__(self) -> Iterator[MeasurementFile]:
        """Iterate over all measurement files"""
        return iter(self.files)
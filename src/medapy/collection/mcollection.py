from typing import Iterator, Union, Tuple, Iterable
from pathlib import Path

from medapy.collection import (MeasurementFile,
                               Polarization,
                               SweepDirection,
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
            if meas_file.check(
                contacts, 
                polarization, 
                sweep_direction, 
                **parameter_filters
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

    def __iter__(self) -> Iterator[MeasurementFile]:
        """Iterate over all measurement files"""
        return iter(self.files)
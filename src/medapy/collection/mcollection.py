from typing import Iterator, Union, Tuple, Iterable
from pathlib import Path

from medapy.collection import (MeasurementFile,
                               ParameterDefinition)
                               

class MeasurementCollection:
    def __init__(self, 
                 collection: str | Path | Iterable[MeasurementFile],
                 parameters: Iterable[ParameterDefinition],
                 file_pattern: str = "*.*",
                 separator: str = "_"):
        
        self._verify_class_in_iterable(parameters, ParameterDefinition, iter_name='parameters')
        self.param_definitions = {param.name_id: param for param in parameters}
        self.separator = separator

        if isinstance(collection, (str, Path)):
            self.folder_path = Path(collection)
            self.files = []
            for f in self.folder_path.glob(file_pattern):
                if f.is_file():
                    self.files.append(MeasurementFile(f, parameters, separator))
            return
        if isinstance(collection, Iterable):
            if not collection:
                self.files = []
                return
            self._verify_class_in_iterable(collection, MeasurementFile, iter_name='collection')
            self.files = list(collection)
            return
        raise ValueError("collection can be str, Path, or Iterable; "
                         f"got {type(collection)}")
    
    def __iter__(self) -> Iterator[MeasurementFile]:
        """Iterate over all measurement files"""
        return iter(self.files)
    
    def __copy__(self):
        """Create a shallow copy of the object"""
        # Create new instance of the same class
        new_obj = type(self)(
            collection=self.files.copy(),
            parameters=list(self.param_definitions.values()))

        # Copy immutable objects directly
        new_obj.file_pattern = self.file_pattern
        new_obj.separator = self.separator
        return new_obj
    
    def __add__(self, other):
        """Enable addition with another Collection or list"""
        if isinstance(other, MeasurementCollection):
            files = self.files + other.files
            parameters = (other.param_definitions | self.param_definitions).values()
            return MeasurementCollection(collection=files,
                                         parameters=parameters)
        raise TypeError(f"Cannot add {type(other)} to MeasurementCollection")
    
    def __len__(self):
        """Return the number of files in collection"""
        return len(self.files)

    def __getitem__(self, index):
        """Enable indexing and slicing"""
        parameters = self.param_definitions.values()
        if isinstance(index, slice):
            return MeasurementCollection(collection=list(self.files[index]),
                                        parameters=parameters)
        return self.files[index]

    def __setitem__(self, index, value):
        """Enable item assignment"""
        if not isinstance(value, MeasurementFile):
            raise TypeError("Can only assign MeasurementFile objects")
        self.files[index] = value

    def __delitem__(self, index):
        """Enable item deletion"""
        del self.files[index]

    def __contains__(self, item):
        """Enable 'in' operator"""
        return item in self.files

    def __str__(self):
        """String representation"""
        res = self.__get_repr_header()
        length = len(self)
        if length <= 60:
            res += self._head_files_str(length)
        else:
            res += self._head_files_str(5)
            res += f'\n{'..':2}    {'...':^8}\n'
            res += self._tail_files_str(5)
        return res
    
    # def __repr__(self):
    #     """Detailed string representation"""
    #     res = self.__get_repr_header()
    #     length = len(self)
    #     if length <= 60:
    #         res += self._head_files_str(length)
    #     else:
    #         res = self._head_files_str(5)
    #         res += f'{'..':2}    {'...':^8}\n'
    #         res += self._tail_files_str(5)
    #     return res
            
    def filter_generator(self, 
               contacts: Union[Tuple[int, int], list[Union[Tuple[int, int], int]], int] = None,
               polarization: str | None = None,
               sweep_direction: str | None = None,
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
               polarization: str | None = None,
               sweep_direction: str | None = None,
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

    def copy(self):
        return self.__copy__()
    
    def append(self, item) -> None:
        """Add a file to collection"""
        if not isinstance(item, MeasurementFile):
            raise TypeError("Can only append MeasurementFile objects")
        self.files.append(item)
    
    def extend(self, iterable: Iterable) -> None:
        """Extend collection from iterable"""
        self._verify_class_in_iterable(iterable, MeasurementFile, iter_name='iterable')
        self.files.extend(iterable)
    
    def pop(self, index: int = -1):
        """
        Remove and return item at index (default last).
        Raises IndexError if collection is empty or index is out of range.
        """
        return self.files.pop(index)
    
    def head(self, n: int = 5) -> None:
        header = self.__get_repr_header()
        print(header + self._head_files_str(n))
    
    def tail(self, n: int = 5) -> None:
        header = self.__get_repr_header()
        print(header + self._tail_files_str(n))
    
    def to_list(self):
        return self.files.copy()
    
    @staticmethod
    def _verify_class_in_iterable(iterable, class_obj, iter_name):
        if not all(isinstance(item, class_obj) for item in iterable):
            raise TypeError(f"All items in {iter_name} must be {class_obj.__name__} objects")
    
    def __get_repr_header(self):
        return f'{'':2}    Filename\n'
    
    def _head_files_str(self, n: int) -> str:
        head = ''
        for (i, f) in enumerate(self.files[:n]):
            head += f'{i:>2}    {f.path.name}\n'
        return head.rstrip('\n')
    
    def _tail_files_str(self, n: str) -> str:
        tail = ''
        ref_idx = len(self.files) - n
        for (i, f) in enumerate(self.files[-n:]):
            tail += f'{ref_idx + i:>2}    {f.path.name}\n'
        return tail.rstrip('\n')
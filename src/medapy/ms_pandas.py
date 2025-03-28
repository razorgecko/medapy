import sys
from pathlib import Path
from collections import Counter
import re
from re import Pattern
import warnings
from typing import Optional, Union

import pandas as pd
from pandas.io.formats.format import DataFrameFormatter
from pandas.io.formats.string import StringFormatter
import pint
from pint.errors import UndefinedUnitError

from .utils.warnings import UnitOverwriteWarning

ureg = pint.UnitRegistry()
pint.set_application_registry(ureg)


def update_column_names(func):
    def wrapper(self, *args, **kwargs):
        # Execute original rename operation
        columns = kwargs['columns']
        invalid_columns = self._get_nonuniques_with_counts(columns.values(), self.labels)
        if invalid_columns:
            raise ValueError(f"New column names conflict with labels: {', '.join(invalid_columns)}")
        res = func(self)(self._obj, *args, **kwargs)

        # Determine which DataFrame to update based on inplace parameter
        df_ms = self if kwargs.get('inplace', False) else res.ms

        # Rename column names in axes, labels and units
        df_ms._update_column_maps(columns)
                    
        return None if kwargs.get('inplace', False) else res
    return wrapper

@pd.api.extensions.register_dataframe_accessor("ms")
class MeasurementSheetAccessor:
    DEFAULT_UNIT_PATTERN = re.compile(r'.*(\(([^\(\)]*)\)|\[([^\[\]]*)\]|\{([^\{\}]*)\})(?!.*[\(\[\{].*[\)\]\}])')
    DEFAULT_UNIT_BRACKETS = '[]'
    DEFAULT_TRANSLATIONS = {}
    DEFAULT_UNIT_FORMAT = '~ms'
    MAIN_AXES = ('x', 'y', 'z')
    DIMENSIONLESS_UNIT = '1'
    
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        
        # Initialize attributes with default values if not present in instance
        if '_ms_labels' not in self._obj.attrs:
            self._obj.attrs['_ms_labels'] = {}
        if '_ms_axes' not in self._obj.attrs:
            self._obj.attrs['_ms_axes'] = {ax: None for ax in self.MAIN_AXES}
        if '_ms_units' not in self._obj.attrs:
            self._obj.attrs['_ms_units'] = {
                col: self.DIMENSIONLESS_UNIT for col in self._obj.columns}
        if '_ms_unit_pattern' not in self._obj.attrs:
            self._obj.attrs['_ms_unit_pattern'] = self.DEFAULT_UNIT_PATTERN
        if '_ms_unit_brackets' not in self._obj.attrs:
            self._obj.attrs['_ms_unit_brackets'] = self.DEFAULT_UNIT_BRACKETS
        if '_ms_translations' not in self._obj.attrs:
            self._obj.attrs['_ms_translations'] = self.DEFAULT_TRANSLATIONS
    
    def __getitem__(self, key: str | list[str]) -> pd.Series | pd.DataFrame:
        """Get column(s) by label(s) or column name(s) using square bracket notation."""
        # Multiple keys
        if isinstance(key, list):
            columns = [self._labels.get(k, k) for k in key]
            self._validate_columns_in_df(columns)
            other_obj = self._obj[columns]
            other_obj.ms.reset_axes()
            other_obj.ms._clear_unused()
            return other_obj
        
        # Single key
        column = self._labels.get(key, key)
        self._validate_columns_in_df(column)
        unit = self.get_unit(column)
        if unit != self.DIMENSIONLESS_UNIT:
            wrapped_unit = self._wrap_unit_in_brackets(unit)
            new_name = f'{column} {wrapped_unit}'
        else:
            new_name = column
        return pd.Series(self._obj[column].to_numpy(), name=new_name)
    
    def __setitem__(self, key: str, value) -> None:
        """Set column(s) by label(s) or column name(s) using square bracket notation."""
        # Single key only yet
        column = self._labels.get(key, key)
        
        # Validate length match
        column_len = self._obj.shape[0]
        try:
            value_len = len(value)
        except TypeError:
            value_len = 1
        if column_len != value_len:
            raise ValueError(f"Length of values ({value_len}) does not match "
                             f"length of measurement sheet ({column_len})")
        
        # Accept pint arrays
        if isinstance(value, pd.Series) and hasattr(value.dtype, 'units'):
            unit = value.dtype.units
            if key in self._obj.columns:
                existing_unit = self.get_unit(key)
                if self._strict_units and unit != existing_unit:
                    raise ValueError(
                        f"Units conflict for column '{column}': "
                        f"assignment object '{type(value)}' unit '{unit}' != "
                        f"existing unit '{existing_unit}'")
            self._obj[column] = value.pint.magnitude
            self.set_unit(column, unit)
            return
        
        self._obj[column] = value
        if column not in self.units:
            self.set_unit(column, '')
        
    def __getattr__(self, name: str) -> pd.Series:
        """Dynamic getter for axes."""
        # Only called if attribute is not found through normal lookup
        if name in self._axes:
            column = self._axes[name]
            return self._obj[column]
        raise AttributeError(f"'{name}' axis is not set")
    
    def __setattr__(self, name, value):
        """Prevent setting axes values manually"""
        # Ensure _obj attribute is set correctly
        if name == '_obj':
            super().__setattr__(name, value)
            return

        if name in self._axes:
            # If name is in _axes, raise error
            raise AttributeError(f"Cannot set value to axis '{name}'")
            
        # For non-_axes attributes, use default behavior
        super().__setattr__(name, value)
    
    def __str__(self):
        index = ['U', 'A', 'L']
        data = dict()
        for column in self._obj.columns.tolist():
            unit = self.get_unit(column) or ''
            axis = self.is_axis(column) or '-'
            labels = self.get_labels(column) or '-'
            
            unit =  self._wrap_unit_in_brackets(unit)
            labels= ', '.join(labels)
            
            data[column] = [unit, axis, labels]
            
        df_ms = pd.DataFrame(data, index=index)
        df_display = pd.concat(objs=[df_ms, self._obj])
        return df_display.__repr__()  
      
    def __repr__(self):
        # Get repr parameters used by pandas
        repr_params = pd.io.formats.format.get_dataframe_repr_params()
        # remove keys unexpected by DataFrameFormatter
        line_width = repr_params.pop('line_width')
        max_colwidth = repr_params.pop('max_colwidth') 

        from pandas import option_context
        
        with option_context("display.max_colwidth", max_colwidth):
            formatter = DataFrameFormatter(frame=self._obj, **repr_params)
            string_formatter = StringFormatter(formatter, line_width=line_width)
            
            # Get original formatted output
            original = string_formatter._get_strcols()
            # Get columns from output and their column number
            columns = [(i, col[0].strip()) for i, col in enumerate(original)]
            
            # Modify data columns
            for i, column in columns[1:]:
                if column == '...':
                    unit = '...'
                    axis = '...'
                    labels = ['...']
                else:
                    unit =  self._wrap_unit_in_brackets(self.get_unit(column))
                    axis = self.is_axis(column) or '-'
                    labels = self.get_labels(column) or '-'
                    
                col_width = len(original[i][0])
                unit_fmt = f"{unit:>{col_width}}"
                axis_fmt = f"{axis:>{col_width}}"
                labels_fmt = f"{' ,'.join(labels):>{col_width}}"
                original[i][1:1] = [unit_fmt, axis_fmt, labels_fmt]
            
            # Modify index column
            idx_width = len(original[0][-1])
            original[0][1:1] = [f"{name:<{idx_width}}" for name in ('U', 'A', 'L')]
                
            # return string_formatter._join_multiline(original)
            return string_formatter.adj.adjoin(1, *original)
    
    @property
    def _units(self) -> dict[str, str]:
        """Access to units -> column mapping"""
        return self._obj.attrs['_ms_units']
    
    @_units.setter
    def _labels(self, value: dict[str, str]) -> None:
        self._obj.attrs['_ms_units'] = value
    
    @property
    def _labels(self) -> dict[str, str]:
        """Access to label -> column mapping"""
        return self._obj.attrs['_ms_labels']

    @_labels.setter
    def _labels(self, value: dict[str, str]) -> None:
        self._obj.attrs['_ms_labels'] = value

    @property
    def _axes(self) -> dict[str, str]:
        """Access to axis -> column mapping"""
        return self._obj.attrs['_ms_axes']

    @_axes.setter
    def _axes(self, value: dict[str, str]) -> None:
        self._obj.attrs['_ms_axes'] = value

    @property
    def _unit_pattern(self) -> Pattern:
        """Access to unit pattern"""
        return self._obj.attrs['_ms_unit_pattern']

    @_unit_pattern.setter
    def _unit_pattern(self, value: Pattern) -> None:
        self._obj.attrs['_ms_unit_pattern'] = value

    @property
    def _unit_brackets(self) -> str:
        """Access to unit brackets format"""
        return self._obj.attrs['_ms_unit_brackets']

    @_unit_brackets.setter
    def _unit_brackets(self, value: str) -> None:
        self._obj.attrs['_ms_unit_brackets'] = value

    @property
    def _translations(self) -> dict[str, str]:
        """Access to unit translations mapping"""
        return self._obj.attrs['_ms_translations']

    @_translations.setter
    def _translations(self, value: dict[str, str]) -> None:
        self._obj.attrs['_ms_translations'] = value
    
    @property
    def axes(self) -> dict[str, str]:
        """Get all axis assignments."""
        return self._axes.copy()
    
    @property
    def units(self) -> dict[str, str]:
        """Get all axis assignments."""
        return self._units.copy()
    
    @property
    def labels(self) -> dict[str, str]:
        """Get all label mappings."""
        return self._labels.copy()
    
    # MeasurementSheet methods
    def init_msheet(
        self,
        units: bool = True,
        *,
        patch_rename: bool = False,
        unit_pattern: str | Pattern | None = None,
        translations: dict[str, str] | None = None,
        brackets: str | None = None,
        format_spec: str | None = None,
        registry_params: dict[str, str] | None = None,
        registry_contexts: str | list[str] | None = None,
        strict_units: bool = True
        ) -> None:
        """
        Initialize measurement sheet with axes and units.

        Parameters
        ----------
        units : bool, default True
            Whether to infer and set units from column names
        patch_rename : bool, default False
            If True, monkey-patches original DataFrame.rename method to update
            column names in labels and axes.
        unit_pattern : str or Pattern, optional
            Pattern to extract units from column names.
            If None, uses the default pattern
        translations : dict[str, str], optional
            Dictionary to set translation of units
            to pint compatible format, e.g. {'Ohm': 'ohm'}
            If None, uses default translations.
        brackets : str, optional
            Brackets to use when wrapping units. Does not affect units parsing.
            Valid options are '()', '[]', '{}'. If None, uses default brackets.
        format_spec : str, optional
            Format to use when converting units to strings.
        registry_params : dict[str, str], optional
            Dictionary to customize pint.UnitRegistry behavior, e.g.
            remove case sensivity when parsing (case_sensitive=False)
        registry_contexts : str or list[str], optional
            One or several pint registry contexts used to convert units.
            For example, context 'Gaussian' is required to convert 
            between 'tesla' and 'oersted'. Names of available contexts 
            can be found inside pint package file `pint/default_en.txt`
        strict_units : bool, default True
            If overwriting units when assigning to existing columns is forbidden
        """
        # Modify unit registry
        if registry_params:
            ureg = pint.UnitRegistry(**registry_params)
            pint.set_application_registry(ureg)
        
        if registry_contexts:
            if isinstance(registry_contexts, str):
                registry_contexts = [registry_contexts]
            ureg.enable_contexts(*registry_contexts)
        
        if patch_rename:
            self._obj.rename = self._patched_rename
        
        self._strict_units = strict_units
        
        # Set unit translations
        self.set_unit_translations(translations)
        
        # Set unit pattern for current sheet
        self.set_unit_pattern(unit_pattern)
        
        # Set brackets to wrap the units
        self.set_unit_brackets(brackets)
        
        # Set format when convert units to strings
        self.set_unit_format(format_spec)
        
        # Set default axes
        self.reset_axes()
        
        # Infer and set units if required
        if units:
            self.infer_ms_units()
        
    def save_msheet(
            self,
            output: str | Path,
            header: bool | list[str] = True,
            float_format: str | None = '%.4f',
            formatter: list[str] | dict[str, str] | None = None,
            index: bool = False,
            brackets: str | None = None,
            unit_format: str | None = None,
            **kwargs
        ) -> None:
        """Save DataFrame with formatted columns to CSV.

        Parameters
        ----------
        output : str or Path
            Path to output file
        header : bool or list of str, default True
            Column headers or True for default headers
        float_format : str, optional
            Format string for float numbers
        formatter : list or dict, optional
            Custom formatters for columns. If list, must match number of columns.
            Format strings should be compatible with str.format() method.
        index : bool, default False
            Whether to write row names
        brackets : str, optional
            Optional brackets for units
        unit_format : str, optional
            Optional unit format string
        **kwargs
            Additional arguments passed to pandas.DataFrame.to_csv()

        Raises
        ------
        ValueError
            If formatter list length doesn't match number of columns
        IOError
            If file cannot be written
        """
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = self._obj.copy(deep=True)
        columns = df.columns

        if isinstance(formatter, list):
            if len(formatter) != len(columns):
                raise ValueError(
                    f"Number of provided formats ({len(formatter)}) does not match "
                    f"number of columns ({len(columns)})"
                )
            formatter = dict(zip(columns, formatter))
        elif isinstance(formatter, dict):
            formatter = {self.get_column(col): fmt for col, fmt in formatter.items()}
        # if header is True:
        #     header = [self._append_unit_to_column(col, brackets, unit_format) for col in columns]
        
        try:
            if formatter is not None:
                self._format_df_inplace(df, formatter)
            df.ms.clear_units(restore_names=True)
            df.to_csv(output_path,
                      float_format=float_format,
                      header=header,
                      index=index,
                      **kwargs)
        except (IOError, OSError) as e:
            raise IOError(f"Failed to save file to {output_path}: {e}")

    def format_msheet(
            self,
            float_format: str | None = '{:.4f}',
            formatter: list[str] | dict[str, str] | None = None
            ) -> None:
        """Format values in measurement sheet according to provided formats.

        Parameters
        ----------
        float_format : str, optional
            Format string for float numbers
        formatter : list or dict, optional
            Custom formatters for columns. If list, must match number of columns.
            Format strings should be compatible with str.format() method ('{:.2f}').
        """
        columns = self._obj.columns
        if not formatter:
            if not float_format:
                raise ValueError("Either 'float_format' or 'formatter' arguments "
                                 f"should be provided: Got '{float_format}' and '{formatter}'")
            formatter = {col: float_format for col in columns}
        elif isinstance(formatter, list):
            if len(formatter) != len(columns):
                raise ValueError(
                    f"Number of provided formats ({len(formatter)}) does not match "
                    f"number of columns ({len(columns)})"
                )
            formatter = dict(zip(columns, formatter))
            
        self._format_df_inplace(self._obj, formatter)
    
    def clear_msheet(self, restore_names: bool = False) -> None:
        """Clear both labels and axis assignments."""
        self.clear_labels()
        self.clear_axes()
        self.clear_units(restore_names)
    
    def infer_ms_units(self,
                       pattern: str | Pattern | None = None,
                       priority: str | None = None,
                       translations: dict[str, str] | None = None
                       ) -> None:
        """
        Infer units from column names and convert to pint dtype.
        
        Parameters
        ----------
        pattern : str or Pattern, optional
            Custom regex pattern to match unit part
        priority : str, optional
            How to resolve conflicts between parsed and existing units:
            - 'name': use units from column names, overwrite existing
            - 'unit': keep existing units, strip column names
            - None: raise error on conflict (default)
        translations : dict[str, str], optional
            Dictionary mapping custom unit names to pint-compatible names.
            These translations take precedence over global translations.
        """
        for col in self._obj.columns:
            self.infer_unit(col, pattern=pattern, priority=priority, translations=translations)
    
    def convert_ms_units(self, to_units: list[str | pint.Unit]) -> None:
        """Convert msheet data to different units at once."""
        columns = self._obj.columns
        if len(to_units) != len(columns):
            raise ValueError(
                f"Number of provided units ({len(to_units)}) is not equal "
                f"to number of columns ({len(columns)})")
        for (col, unit) in zip(columns, to_units):
            self.convert_unit(col, unit)
    
    def get_ms_state(self) -> dict:
        """Get MSheet state as dict"""
        return self._get_ms_state_of_df(self._obj)
    
    def set_ms_state(self, attrs) -> None:
        """Set MSheet state from dict"""
        self._obj.attrs.update(attrs)
    
    # Unit methods
    def infer_unit(self, column: str,
                   pattern: str | Pattern = None,
                   priority: str | None = None,
                   translations: dict[str, str] | None = None
                   ) -> None:
        """
        Infer and set unit for a column while preserving column order.

        Parameters
        ----------
        column : str
            Column name
        pattern : str or Pattern, optional
            Custom regex pattern to match unit part
        priority : str, optional
            How to resolve conflicts between parsed and existing units:
            - 'name': use units from column names, overwrite existing
            - 'unit': keep existing units, strip column names
            - None: raise error on conflict (default)
        translations : dict[str, str], optional
            Dictionary mapping custom unit names to pint-compatible names.
            These translations take precedence over global translations.
        """
        if priority not in ['name', 'unit', None]:
            raise ValueError("priority must be 'name', 'unit' or None")

        column = self.get_column(column)
        existing_unit = self.get_unit(column)
        parsed_unit = self.parse_unit(column, pattern)
        new_col = self.strip_unit(column, pattern) if parsed_unit is not None else column
        
        # Translate custom unit name to pint-compatible. If None, returns None
        parsed_unit = self._translate_unit(parsed_unit, translations)
        # No unit in column name (None) - set dimensionless
        parsed_unit = '' if parsed_unit is None else parsed_unit
        # Convert to pint.Unit to compare with existing unit
        parsed_unit = ureg.Unit(parsed_unit)

        # No conflict: No existing unit - set parsed unit and rename column
        if existing_unit == self.DIMENSIONLESS_UNIT:
            self.set_unit(column, parsed_unit)
            self.rename(columns={column: new_col})
            return

        # No conflict: Existing and parsed units are equal - rename column
        if parsed_unit == existing_unit:
            self.rename(columns={column: new_col})
            return
        
        # Unit conflict: give priority to unit over dimensionless
        if parsed_unit == self.DIMENSIONLESS_UNIT:
            self.rename(columns={column: new_col})
            return
        
        # Handle unit conflict
        compatible = ureg.is_compatible_with(parsed_unit, existing_unit)
        if priority is None:
            raise ValueError(
                f"Unit conflict for column '{column}': "
                f"name unit '{parsed_unit}' != existing unit '{existing_unit}'. "
                f"Units are {'compatible' if compatible else 'not compatible'}"
            )

        if priority == 'name':
            message = (
                f"Column '{column}' units '{existing_unit}' overwritten with "
                f"'{parsed_unit}' from column name. Units are "
                f"{'compatible' if compatible else 'not compatible'}"
            )
            warnings.warn(message, UnitOverwriteWarning)
            self.set_unit(column, parsed_unit)
        else:  # priority == 'unit'
            message = (
                f"Keeping existing unit '{existing_unit}' for column '{column}' "
                f"instead of '{parsed_unit}' from column name. Units are "
                f"{'compatible' if compatible else 'not compatible'}"
            )
            warnings.warn(message, UnitOverwriteWarning)
      
        self.rename(columns={column: new_col})
    
    def get_unit(self, column: str) -> str | None:
        """
        Get unit from a column.
        
        Returns:
        - str if column has unit
        - None if no unit found
        """
        column = self.get_column(column)

        unit = self._units.get(column)
        return unit
    
    def set_unit(self, column: str, unit: Union[str, pint.Unit, None] = None) -> None:
        """
        Set unit for a column without converting values.
        Keeps the original values intact while changing only the unit.

        Parameters
        ----------
        column : str
            Column name
        unit : str or pint.Unit, or None
            Unit to set. None and '' set unit to dimensionless
        """
        column = self.get_column(column)
        
        # Set new unit without conversion
        # If unit = None, make dimensionless
        unit = '' if unit is None else unit
        unit = self._validate_unit(unit)
        unit_str = str(unit)
            
        if unit_str in ('', 'dimensionless', '1'):
            unit_str = self.DIMENSIONLESS_UNIT
        self._units[column] = unit_str
    
    def convert_unit(self,
                     column: str,
                     to_unit: Union[str, pint.Unit],
                     contexts: str | list[str] | None = None
                     ) -> None:
        """Convert column data to different unit."""
        column = self.get_column(column)
        unit = self.get_unit(column)
        unit = ureg(unit)
        to_unit = self._validate_unit(to_unit)
        
        if contexts:
            if isinstance(contexts, str):
                contexts = [contexts]
            with ureg.context(*contexts):
                coef = unit.to(to_unit).m # magnitude of pint.Quantity
        else:
            coef = unit.to(to_unit).m # magnitude of pint.Quantity
            
        self._obj[column] *= coef
        self.set_unit(column, to_unit)
    
    def parse_unit(self, column: str, pattern: Optional[Union[str, Pattern]] = None) -> Optional[str]:
        """
        Parse unit from column name using regex pattern.

        Parameters
        ----------
        column : str
            Column name to parse unit from
        pattern : str or Pattern, optional
            Custom regex pattern. If None, uses self.UNIT_PATTERN

        Returns
        -------
        Optional[str]
            Unit string if found, empty string for dimensionless, None if no unit marking
        """
        if pattern is not None:
            use_pattern = self._compile_pattern(pattern)
        else:
            use_pattern = self._unit_pattern
        match = use_pattern.match(column)
        if not match:
            return None

        # Get the unit part from whichever bracket type matched
        unit_str = next((group for group in match.groups()[1:] if group is not None), None)
        if unit_str is None:
            return None

        # Handle dimensionless cases
        if unit_str.strip() in {'', 'a.u.', 'a. u.'}:
            return self.DIMENSIONLESS_UNIT

        return unit_str
    
    def strip_unit(self, column: str, pattern: Optional[Union[str, Pattern]] = None) -> str:
        """
        Remove unit part from column name.

        Parameters
        ----------
        column : str
            Column name with unit in brackets (any of (), [], {})
        pattern : str or Pattern, optional
            Custom regex pattern to match unit part. If None, uses default pattern
            that matches last bracketed expression

        Returns
        -------
        str
            Column name without unit part

        Examples
        --------
        >>> df.ms.strip_unit('temperature(degC)')
        'temperature'
        >>> df.ms.strip_unit('custom[kg/m^2]')
        'custom'
        """
        if pattern is not None:
            use_pattern = self._compile_pattern(pattern)
        else:
            use_pattern = self._unit_pattern
        match = use_pattern.match(column)
        if match:
            return column.replace(match.group(1), '').strip()
        return column
    
    def restore_unit(self, column: str,
                    brackets: Optional[str] = None,
                    format_spec: Optional[str] = None) -> None:
        """
        Append unit wrapped in brackets to column name.

        Parameters
        ----------
        column : str
            Column name
        brackets : str, optional
            Custom brackets to wrap unit. If None, uses default brackets
        format_spec : str, optional
            Format to use when convert unit to string. If None, uses default format
        """
        new_col = self._append_unit_to_column(column, brackets, format_spec)
        self.rename(columns={column: new_col})
        return new_col
    
    def clear_units(self, restore_names: bool = False) -> None:
        """Clear all column units."""
        for col in self._obj.columns:
            if restore_names:
                new_col = self.restore_unit(col)
                self.set_unit(new_col, None)
                continue
            self.set_unit(col, None)    
    
    def wu(self, name: str) -> pd.Series:
        """Get column values as series with pint unit"""
        if 'sys' not in sys.modules:
            raise RuntimeError("Required module 'pint_pandas' is not loaded")
        
        column = self.get_column(name)
        series = self._obj[column]
        unit = self.get_unit(column)
        return pd.Series(series, dtype=f'pint[{unit}]')
    
    # Label methods
    def get_column(self, name: str) -> str:
        """Get original column name from either label or column name.

        Args:
            name: Label or column name

        Returns:
            Original column name

        Raises:
            KeyError: If name is neither a label nor a column
        """
        if name in self._labels:
            return self._labels[name]
        elif name in self._obj.columns:
            return name
        else:
            raise KeyError(f"'{name}' is neither a label nor a column")
    
    def rename(self, columns: dict[str, str]) -> None:
        self._obj.rename(columns=columns, inplace=True)
        self._update_column_maps(columns)
    
    def add_labels(self, labels: dict[str, str]) -> None:
        invalid_cols = set(labels.keys()) - set(self._obj.columns)
        if invalid_cols:
            raise ValueError(f"Invalid column names: {', '.join(invalid_cols)}")
        
        labels_dict = dict()
        for column, label in labels.items():
            if not isinstance(label, str):
                for lbl in label:
                    labels_dict[lbl] = column
            else:
                labels_dict[label] = column
            
        invalid_labels = self._get_nonuniques_with_counts(self._obj.columns, labels_dict)
        if invalid_labels:
            raise ValueError(f"Labels conflict with existing columns: {', '.join(invalid_labels)}")
        
        self._labels.update(labels_dict)
      
    def add_label(self, column: str, label: str) -> None:
        """Add a label for a column."""
        if label in self._obj.columns:
            raise ValueError(f"Label '{label}' conflicts with existing column name")
        if column not in self._obj.columns:
            raise ValueError(f"Column '{column}' does not exist")
        self._labels[label] = column
        
    def rename_label(self, old: str, new: str) -> None:
        """Rename a label while preserving its column mapping.

        Args:
            old: Existing label name
            new: New label name

        Raises:
            KeyError: If old doesn't exist
            ValueError: If new conflicts with existing column name
        """
        if old not in self._labels:
            raise KeyError(f"Label '{old}' not found")

        if new in self._obj.columns:
            raise ValueError(f"New label '{new}' conflicts with existing column name")

        if new in self._labels:
            raise ValueError(f"Label '{new}' already exists")

        # Add new label with same column mapping
        self._labels[new] = self._labels.pop(old)

    def remove_label(self, label: str) -> None:
        """Remove a label mapping."""
        if label not in self._labels:
            raise KeyError(f"Label '{label}' not found")
        self._labels.pop(label, None)
    
    def get_labels(self, column: str) -> tuple[str]:
        """Get all labels assigned to column"""
        return tuple(lbl for lbl, col in self._labels.items() if col == column)

    
    def clear_labels(self) -> None:
        """Clear all label mappings."""
        self._labels.clear()

    # Axes methods
    def reset_axes(self) -> None:
        columns = self._obj.columns
        if len(columns) > 0:
            self._axes['x'] = columns[0]
        if len(columns) > 1:
            self._axes['y'] = columns[1]
        if len(columns) > 2:
            self._axes['z'] = columns[2] 
    
    def is_axis(self, name: str) -> str | None:
        """Check if column or label is assigned to any axis.

        Args:
            name: Column name or label to check

        Returns:
            Axis name ('x', 'y', 'z', etc.) if assigned, None otherwise
        """
        # Resolve column name if a label was passed
        column = self.get_column(name)

        return next((axis for axis, col in self._axes.items() 
                    if col == column), None)
    
    def set_as_axis(self, name: Union[str, None], axis: str, swap: bool = False) -> None:
        """Set column as a named axis or remove axis assignment if name is None.
        
        Args:
            name: Column name, label, or None to remove assignment
            axis: Name of the axis ('x', 'y', 'z', etc.)
            swap: If True and column is already assigned to another axis,
                swap the axis assignments. If False, remove the previous
                axis assignment.
        """
        if name is None:
            self.remove_axis(axis)
            return

        # Resolve column name if a label was passed
        column = self.get_column(name)

        # Check if column is already assigned to another axis
        existing_axis = self.is_axis(column)
        
        if existing_axis == axis:
            return
        
        if existing_axis:
            if swap:
                # Swap axis assignments
                old_column = self._axes.get(axis)
                if old_column:
                    self._axes[existing_axis] = old_column
                else:
                    self._axes.pop(existing_axis)
            else:
                # Remove previous axis assignment
                self._axes.pop(existing_axis)

        self._axes[axis] = column
    
    def set_as_x(self, column: Union[str, None], swap: bool = False) -> None:
        """Set column as x axis"""
        self.set_as_axis(column, 'x', swap)
        
    def set_as_y(self, column: Union[str, None], swap: bool = False) -> None:
        """Set column as y axis"""
        self.set_as_axis(column, 'y', swap)
    
    def set_as_z(self, column: Union[str, None], swap: bool = False) -> None:
        """Set column as z axis"""
        self.set_as_axis(column, 'z', swap)
    
    def remove_axis(self, axis: str) -> None:
        """Remove an axis assignment."""
        if axis not in self._axes:
            raise KeyError(f"Axis '{axis}' not set")
        if axis in self.MAIN_AXES:
            self._axes[axis] = None
        else:
            self._axes.pop(axis)
    
    def remove_x(self) -> None:
        """Remove x axis assignment."""
        self.remove_axis('x')
    
    def remove_y(self) -> None:
        """Remove y axis assignment."""
        self.remove_axis('y')
    
    def remove_z(self) -> None:
        """Remove z axis assignment."""
        self.remove_axis('z')
    
    def clear_axes(self) -> None:
        """Clear all axis assignments."""
        self._axes.clear()
        self.reset_axes()

    # DataFrame operations
    def concat(self,
               objs: pd.DataFrame | list[pd.DataFrame],
               *,
               drop_x: bool = True
               ) -> pd.DataFrame:
        if isinstance(objs, pd.DataFrame):
            objs = [objs]
        if drop_x:
            objs = [obj.copy(deep=True) for obj in objs]
            for obj in objs:
                x_axis = obj.attrs.get('_ms_axes', {}).get('x')
                if x_axis:
                    obj.drop(columns=x_axis, inplace=True)
                try:
                    obj.ms._clear_unused()
                except Exception as e:
                    raise Exception(e)
        
        all_columns = [self._obj.columns.tolist()]
        all_labels = [self.labels]
        all_units = [self.units]
        for obj in objs:
            columns = obj.columns.tolist()
            all_columns.append(columns)
            
            labels = obj.attrs.get('_ms_labels', {})
            all_labels.append(labels)
            
            units = obj.attrs.get('_ms_units', {})
            if not units:
                units = {col: self.DIMENSIONLESS_UNIT for col in columns}
            all_units.append(units)

        columns_overlap = self._get_nonuniques_with_counts(*all_columns)
        if columns_overlap:
            raise ValueError("Column names of concatenating objects have duplicates "
                             f"(name: times): {str(columns_overlap)[1:-1]}")
        
        labels_overlap = self._get_nonuniques_with_counts(*map(dict.keys, all_labels))
        if labels_overlap:
            raise ValueError("Labels of concatenating objects have duplicates "
                             f"(name: times): {str(labels_overlap)[1:-1]}")
        
        df = pd.concat(objs=[self._obj, *objs], axis='columns')
        ms_state = self.get_ms_state()
        self._set_ms_state_to_df(df, ms_state)
        merged_labels = dict()
        for d in all_labels:
            merged_labels.update(d)
        df.attrs['_ms_labels'] = merged_labels.copy()
        
        merged_units = dict()
        for d in all_units:
            merged_units.update(d)
        df.attrs['_ms_units'] = merged_units
        
        return df
    
    # Unit customization methods
    def set_unit_translations(self, translations: Optional[dict[str, str]] = None) -> None:
        """
        Set global unit translations

        Parameters
        ----------
        translations : dict[str, str], optional
            Dictionary mapping custom unit names to pint-compatible names.
            For example, {'Ohm': 'ohm'}. If None, resets to default.
        """
        if translations is None:
            self._translations = self.DEFAULT_TRANSLATIONS
            return
        
        self._validate_translations(translations)
        self._translations = translations.copy()
        
    def set_unit_brackets(self, brackets: Optional[str]) -> None:
        """
        Set global unit brackets
        
        Parameters
        ----------
        brackets : str, optional
            Brackets to use when wrapping units. Does not affect units parsing.
            Valid options are '()', '[]', '{}'. If None, uses default brackets.
        """
        if brackets is None:
            self._unit_brackets = self.DEFAULT_UNIT_BRACKETS
            return

        self._validate_brackets(brackets)
        self._unit_brackets = brackets

    def set_unit_format(self, format_spec: Optional[str] = None) -> None:
        """
        Set global unit format specification for pint

        Parameters
        ----------
        format_spec : str, optional
            Format specification for pint unit string conversion.
            For example: '~P' for pretty format, '~L' for latex.
            If None, resets to default (custom '~ms' format)

        Raises
        ------
        ValueError
            If format specification is invalid
        """
        if format_spec is None:
            ureg.formatter.default_format = self.DEFAULT_UNIT_FORMAT
            self._unit_format = self.DEFAULT_UNIT_FORMAT
            return

        self._validate_format_spec(format_spec)
        ureg.formatter.default_format = format_spec
    
    def set_unit_pattern(self, pattern: Optional[Union[str, Pattern]] = None) -> None:
        """
        Set default unit pattern for current measurement sheet.
        
        Parameters
        ----------
        pattern : str or Pattern, optional
            Pattern to set. If None, resets to default pattern
        """
        if pattern is not None:
            self._unit_pattern = self._compile_pattern(pattern)
        else:
            self._unit_pattern = self.DEFAULT_UNIT_PATTERN
    
    # Protected methods
    def _set_column_states(self, columns, values=None, units=None, axes=None, labels=None) -> None:
        """
        Set values and metadata for a dataframe columns.
        Helper method to use with other accessors 

        Parameters
        ----------
        columns : str
            Name of the columns to modify.
        values : list[array-like], optional
            Values to assign to the columns.
        units : list[str], optional
            Units to set for the columns.
        axes : list[str], optional
            Axes to set for the columns.
        labels : list[str], optional
            Labels to add to the columns.

        Returns
        -------
        None

        Notes
        -----
        Only performs operations for non-None parameters.
        """
        n_cols = len(columns)
        
        # Set default lists if None
        if values is None:
            values = [None] * n_cols
        if units is None:
            units = [None] * n_cols
        if axes is  None:
            axes = [None] * n_cols
        if labels is None:
            labels = [None] * n_cols
            
        for (col, val, unt, axs, lbl) in zip(columns, values, units, axes, labels):
            self._set_column_state(column=col, values=val, unit=unt, axis=axs, label=lbl)
    
    def _set_column_state(self, column, values=None, unit=None, axis=None, label=None) -> None:
        """
        Set values and metadata for a dataframe column.
        Helper method to use with other accessors 

        Parameters
        ----------
        column : str
            Name of the column to modify.
        values : array-like, optional
            Values to assign to the column.
        unit : str, optional
            Unit to set for the column.
        axis : str, optional
            Axis to set for the column.
        label : str, optional
            Label to add to the column.

        Returns
        -------
        None

        Notes
        -----
        Only performs operations for non-None parameters.
        """
        # Set values if provided
        if values is not None:
            self[column] = values

        if unit is not None:
            self.set_unit(column, unit)

        if axis is not None:
            self.set_as_axis(column, axis)

        if label is not None:
            self.add_labels({column: label})
    
    def _get_nonuniques_with_counts(self, *sequences):
        items = [item for seq in sequences for item in seq]
        counts = Counter(items)
        duplicates = {item: count for item, count in counts.items() if count > 1}
        return duplicates
        
    def _make_empty_replica_len(self, **kwargs) -> pd.DataFrame:
        empty_df = pd.DataFrame(**kwargs)
        attrs = self.get_ms_state()
        for attr in ('_ms_labels', '_ms_axes', '_ms_units'):
            attrs.pop(attr)
        self._set_ms_state_to_df(empty_df, attrs)
        return empty_df
    
    def _translate_unit(self,
                        unit: Optional[str],
                        translations: Optional[dict[str, str]] = None) -> Optional[str]:
        """Translate unit string using provided or global translations"""
        if unit is None:
            return None

        # Combine global and method-specific translations
        # Method-specific translations take precedence
        all_translations = self._translations.copy()
        if translations:
            all_translations.update(translations)
        for u_incorrect, u_correct in all_translations.items():
            unit = unit.replace(u_incorrect, u_correct)
        return unit
    
    def _compile_pattern(self, pattern: Union[str, Pattern]) -> Pattern:
        """
        Compile pattern if it's a string, otherwise return as is.

        Parameters
        ----------
        pattern : str or Pattern
            Pattern to compile

        Returns
        -------
        Pattern
            Compiled regular expression pattern

        Raises
        ------
        TypeError
            If pattern is neither string nor compiled regex
        ValueError
            If string pattern is invalid
        """
        if isinstance(pattern, Pattern):
            return pattern
        if isinstance(pattern, str):
            try:
                return re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regular expression pattern: {e}")
        raise TypeError(f"Pattern must be string or compiled regex, got {type(pattern)}")
            
    def _append_unit_to_column(self, column: str,
                    brackets: Optional[str] = None,
                    format_spec: Optional[str] = None) -> str:
        """
        Append unit wrapped in brackets to column name.

        Parameters
        ----------
        column : str
            Column name
        brackets : str, optional
            Custom brackets to wrap unit. If None, uses default brackets
        format_spec : str, optional
            Format to use when convert unit to string. If None, uses default format
        """
        unit = self.get_unit(column)
        wrapped_unit = self._wrap_unit_in_brackets(unit, brackets, format_spec)
        return f'{column} {wrapped_unit}'

    def _wrap_unit_in_brackets(self, unit, brackets=None, format_spec=None) -> str:
        if brackets is not None:
            self._validate_brackets(brackets)
            left, right = brackets
        else:
            left, right = self._unit_brackets
            
        if format_spec is not None:
            self._validate_format_spec(format_spec)
            use_format = format_spec
        else:
            use_format = ureg.formatter.default_format
        unit = self._validate_unit(unit)
        formatted_unit = format(unit, use_format)
        return f'{left}{formatted_unit}{right}'
    
    def _update_column_maps(self, columns: dict[str, str]) -> None:
        """Remap axes, labels, and units when columns change names"""
        # Use dict copies (self.units, self.labels, self.axes) 
        # to avoid changing keys during iteration
        for old_name, new_name in columns.items():
            # Update axes dictionary (unique mapping)
            for ax, col in self.axes.items():
                if col == old_name:
                    self._obj.attrs['_ms_axes'][ax] = new_name
                    break

            # Update labels dictionary (non-unique mapping)
            for label, col in self.labels.items():
                if col == old_name:
                    self._obj.attrs['_ms_labels'][label] = new_name
                    
            # Update units dictionary (unique mapping)
            for col in self.units:
                if col == old_name:
                    self._obj.attrs['_ms_units'][new_name] = self._obj.attrs['_ms_units'].pop(old_name)
    
    def _clear_unused(self) -> None:
        """Clear axes, labels and units for columns that are not in df"""
        for axis, column in self.axes.items():
            if column not in self._obj.columns:
                self.remove_axis(axis)
        
        for label, column in self.labels.items():
            if column not in self._obj.columns:
                self.remove_label(label)
        
        for column, unit in self.units.items():
            if column not in self._obj.columns:
                self._units.pop(column)
    
    @staticmethod
    def _get_ms_state_of_df(df: pd.DataFrame) -> dict:
        attrs = dict()
        df_attrs = df.attrs.copy()
        for key, value in df_attrs.items():
            if isinstance(key, str) and key.startswith('_ms_'):
                attrs[key] = value
        return attrs
    
    @staticmethod
    def _set_ms_state_to_df(df: pd.DataFrame, attrs: dict) -> None:
        for key in attrs:
            if not (isinstance(key, str) and key.startswith('_ms_')):
                raise ValueError(f"Unsupported key {key}")
        df.attrs.update(attrs)
    
    @staticmethod
    def _format_df_inplace(df: pd.DataFrame, formatter: dict[str, str]) -> None:
        """Format DataFrame columns in-place using provided formatters.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to format
        formatter : dict
            Dictionary mapping column names to format strings

        Raises
        ------
        ValueError
            If formatter contains invalid column names

        Notes
        -----
        The DataFrame is modified in-place. Format strings should be compatible
        with str.format() method.
        """
        invalid_cols = set(formatter) - set(df.columns)
        if invalid_cols:
            raise ValueError(f"Invalid column names in formatter: {invalid_cols}")            
    
        # Handle columns with units
        for col, fmt in formatter.items():
            df[col] = df[col].map(fmt.format)
    
    @update_column_names
    def _patched_rename(self, *args, **kwargs):
        """Handle column rename events"""
        return pd.DataFrame.rename

    # Validation methods
    def _validate_translations(self, translations: dict[str, str]) -> None:
        """
        Validate that all target units in translations are pint-compatible.

        Parameters
        ----------
        translations : dict[str, str]
            Dictionary mapping custom unit names to pint-compatible names

        Raises
        ------
        ValueError
            If any target unit is not recognized by pint
        """
        
        invalid_units = []
        for source, target in translations.items():
            try:
                ureg.Unit(target)
            except (pint.UndefinedUnitError, pint.DimensionalityError):
                invalid_units.append((source, target))

        if invalid_units:
            units_str = ', '.join(f"'{s}->{t}'" for s, t in invalid_units)
            raise ValueError(
                f"Following translations contain invalid target units: {units_str}"
            )
    
    def _validate_brackets(self, brackets: str) -> None:
        """
        Validate that string contains exactly one valid pair of brackets.

        Parameters
        ----------
        brackets : str
            String containing a pair of brackets to check

        Raises
        ------
        TypeError
            If brackets parameter is not a string
        ValueError
            If brackets are not a valid pair
        """
        if not isinstance(brackets, str):
            raise TypeError("Input must be a string")

        valid_pairs = {
            '(': ')',
            '[': ']',
            '{': '}'
        }

        if len(brackets) != 2:
            raise ValueError(
                f"Expected exactly 2 characters, got {len(brackets)}"
            )

        left, right = brackets

        if left not in valid_pairs:
            raise ValueError(
                f"Invalid opening bracket '{left}'. Expected one of: {', '.join(valid_pairs.keys())}"
            )

        if right != valid_pairs[left]:
            raise ValueError(
                f"Mismatched brackets: '{left}{right}'. Expected '{left}{valid_pairs[left]}'"
            )
    
    def _validate_format_spec(self, format_spec: str) -> None:
        """
        Validate that format specification is compatible with pint.

        Parameters
        ----------
        format_spec : str
            Format specification to validate

        Raises
        ------
        ValueError
            If format specification is invalid
        """
        # Create a test unit to validate format
        test_unit = ureg.Unit('meter')
        try:
            format(test_unit, format_spec)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid format specification '{format_spec}'. Error: {e}"
            )
    
    def _validate_columns_in_df(self, columns: str | list[str]):
        """Validate all columns exist"""
        # Check single column
        if isinstance(columns, str):
            if columns not in self._obj.columns:
                raise KeyError(f"'{columns}' is neither a valid column nor a label")
            return
        
        invalid = [col for col in columns if col not in self._obj.columns]
        if invalid:
            raise KeyError("Following names are neither valid columns nor labels: "
                           f"{', '.join(invalid)}")
            
    def _validate_unit(self, unit: str) -> pint.Unit:
        """Validate the unit is defined in pint registry"""
        if isinstance(unit, pint.Unit):
            return unit

        try:
            pint_unit = ureg.Unit(unit)
        except UndefinedUnitError:
            raise ValueError(f"'{unit}' is not defined in pint unit registry")
        return pint_unit
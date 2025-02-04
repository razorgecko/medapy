import re
import warnings
from pathlib import Path
from re import Pattern
from typing import Optional, Union

import numpy.typing as npt
import pandas as pd
import pint
from pint_pandas import PintType

from .utils.warnings import UnitOverwriteWarning


ureg = PintType.ureg

def update_column_names(func):
    def wrapper(self, *args, **kwargs):
        # Execute original rename operation
        columns = kwargs['columns']
        res = func(self)(self._obj, *args, **kwargs)

        # Determine which DataFrame to update based on inplace parameter
        target_df = self._obj if kwargs.get('inplace', False) else res

        # Update axes and labels
        for old_name, new_name in columns.items():
            # Update axes dictionary (unique mapping)
            for ax, val in target_df.attrs['_ms_axes'].items():
                if val == old_name:
                    target_df.attrs['_ms_axes'][ax] = new_name
                    break

            # Update labels dictionary (non-unique mapping)
            for label, val in target_df.attrs['_ms_labels'].items():
                if val == old_name:
                    target_df.attrs['_ms_labels'][label] = new_name
                    
        return None if kwargs.get('inplace', False) else res
    return wrapper

@pd.api.extensions.register_dataframe_accessor("ms")
class MeasurementSheetAccessor:
    DEFAULT_UNIT_PATTERN = re.compile(r'.*(\(([^\(\)]*)\)|\[([^\[\]]*)\]|\{([^\{\}]*)\})(?!.*[\(\[\{].*[\)\]\}])')
    DEFAULT_UNIT_BRACKETS = '[]'
    DEFAULT_TRANSLATIONS = {}
    DEFAULT_UNIT_FORMAT = '~ms'
    MAIN_AXES = ('x', 'y', 'z')
    
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        
        # Initialize attributes with default values if not present in instance
        if '_ms_labels' not in self._obj.attrs:
            self._obj.attrs['_ms_labels'] = {}
        if '_ms_axes' not in self._obj.attrs:
            self._obj.attrs['_ms_axes'] = {}
        if '_ms_unit_pattern' not in self._obj.attrs:
            self._obj.attrs['_ms_unit_pattern'] = self.DEFAULT_UNIT_PATTERN
        if '_ms_unit_brackets' not in self._obj.attrs:
            self._obj.attrs['_ms_unit_brackets'] = self.DEFAULT_UNIT_BRACKETS
        if '_ms_translations' not in self._obj.attrs:
            self._obj.attrs['_ms_translations'] = self.DEFAULT_TRANSLATIONS
    
    def __getitem__(self, key: str | list[str]) -> pd.Series | pd.DataFrame:
        """Get column(s) by label(s) or column name(s) using square bracket notation."""
        if isinstance(key, (list, tuple)):
            columns = [self._labels.get(k, k) for k in key]
            # Validate all columns exist
            invalid = [col for col in columns if col not in self._obj.columns]
            if invalid:
                raise KeyError(f"Following names are neither valid columns nor labels: {invalid}")
            return self._obj[columns]
        
        # Single key
        column = self._labels.get(key, key)
        if column not in self._obj.columns:
            raise KeyError(f"'{key}' is neither a valid column nor a label")
        return self.get(key)

    def __getattr__(self, name: str) -> pd.Series:
        """Dynamic getter for axes."""
        if name in self._axes:
            column = self._axes[name]
            return self._obj[column].to_numpy(dtype=float)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __repr__(self) -> str:
        """Return string representation of the accessor's state."""
        lines = []

        # Add header section
        lines.append("Columns:")
        for i, col in enumerate(self._obj.columns):
            unit = self.get_unit(col)
            lines.append(f"  {i}: {col} [{unit}]")

        # Add labels section if any labels exist
        if self._labels:
            lines.append("\nLabels:")
            for label, column in self._labels.items():
                lines.append(f"  {label} -> {column}")
        else:
            lines.append("\nLabels: none")

        # Add axes section if any axes are assigned
        if self._axes:
            lines.append("\nAxes:")
            for axis, column in self._axes.items():
                lines.append(f"  {axis} -> {column}")
        else:
            lines.append("\nAxes: none")

        return "\n".join(lines)
    
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
    def x(self) -> pd.Series:
        if 'x' not in self._axes:
            raise ValueError("x axis not set")
        return self._obj[self._axes['x']].to_numpy(dtype=float)
    
    @property
    def y(self) -> pd.Series:
        if 'y' not in self._axes:
            raise ValueError("y axis not set")
        return self._obj[self._axes['y']].to_numpy(dtype=float)

    @property
    def z(self) -> pd.Series:
        if 'z' not in self._axes:
            raise ValueError("z axis not set")
        return self._obj[self._axes['z']].to_numpy(dtype=float)
    
    @property
    def axes(self) -> dict[str, str]:
        """Get all axis assignments."""
        return self._axes.copy()
    
    @property
    def units(self) -> dict[str, str]:
        """Get all axis assignments."""
        units = {col: self.get_unit(col) for col in self._obj.columns}
        return units
    
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
        registry_params: dict[str, str] | None = None
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
        """
        # Modify unit registry
        if registry_params:
            ureg = pint.UnitRegistry(**registry_params)
            pint.set_application_registry(ureg)
            # self._ureg = pint.get_application_registry()
        
        if patch_rename:
            self._obj.rename = self._patched_rename
        
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
        #     header = [self._append_unit(col, brackets, unit_format) for col in columns]
        
        try:
            if formatter is not None:
                self._format_df_inplace(df, formatter)
                df.ms.clear_units(restore_names=True)
                df.to_csv(output_path, header=header, index=index, **kwargs)
            else:
                df.ms.clear_units(restore_names=True)
                df.to_csv(
                    output_path,
                    float_format=float_format,
                    header=header,
                    index=index,
                    **kwargs
                )
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
        if not existing_unit:
            self.set_unit(column, parsed_unit)
            self.rename_column(column, new_col)
            return

        # No conflict: Existing and parsed units are equal - rename column
        if parsed_unit == existing_unit:
            self.rename_column(column, new_col)
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

        self.rename_column(column, new_col)
    
    def get_unit(self, column: str, parse: bool = False) -> Union[pint.Unit, str, None]:
        """
        Get unit from a column. Returns:
        - pint.Unit if column has pint dtype
        - str if unit can be parsed from column name
        - None if no unit found
        """
        column = self.get_column(column)

        series = self._obj[column]
        if hasattr(series.dtype, 'units'):
            return series.dtype.units
        
        if parse:
            return self.parse_unit(column)
    
    def set_unit(self, column: str, unit: Union[str, pint.Unit, None] = None) -> None:
        """
        Set unit for a column without converting values.
        Keeps the original values intact while changing only the unit.

        Parameters
        ----------
        column : str
            Column name
        unit : str or pint.Unit, or None
            Unit to set. If None, removes unit
        """
        column = self.get_column(column)

        # Get raw values if column already has pint dtype
        series = self.get(column, raw=True)
        
        # Set new unit without conversion
        # If unit = None, remove units
        if unit is not None:
            self._obj[column] = series.astype(f'pint[{unit}]')
        else:
            self._obj[column] = series
    
    def convert_unit(self, column: str, to_unit: Union[str, pint.Unit], context: Optional[str] = None) -> None:
        """Convert column data to different unit."""
        column = self.get_column(column)
        if not hasattr(self._obj[column].dtype, 'units'):
            raise ValueError(f"Column '{column}' does not have units")
        self._obj[column] = self._obj[column].pint.to(to_unit)
    
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
        if unit_str.strip() in {'1', '', ' '}:
            return ''

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
        new_col = self._append_unit(column, brackets, format_spec)
        self.rename_column(column, new_col)
        return new_col
    
    def clear_units(self, restore_names: bool = False) -> None:
        """Clear all column units."""
        for col in self._obj.columns:
            if restore_names:
                new_col = self.restore_unit(col)
                self.set_unit(new_col, None)
                continue
            self.set_unit(col, None)    
    
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
    
    # Label methods
    def get(self, name: str, raw: bool = False) -> pd.Series:
        """Get data by column name or label."""
        column = self.get_column(name)
        series = self._obj[column]
        if raw:
            if hasattr(series.dtype, 'units'):
                return series.pint.magnitude
        return series

    def replace_values(self, name: str, values: npt.ArrayLike) -> None:
        """Manually replace values of the column."""
        column = self.get_column(name)
        unit = self.get_unit(column)
        self._obj[column] = pd.Series(values, dtype=f'pint[{unit}]')
    
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
    
    def rename_column(self, old: str, new: str) -> None:
        if old != new:
            self._obj.rename(columns={old: new}, inplace=True)
    
    def add_label(self, label: str, column: str) -> None:
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

        # Store the column mapping and remove old label
        column = self._labels.pop(old)

        # Add new label with same column mapping
        self._labels[new] = column

    def remove_label(self, label: str) -> None:
        """Remove a label mapping."""
        if label not in self._labels:
            raise KeyError(f"Label '{label}' not found")
        self._labels.pop(label, None)
    
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
    
    def is_axis(self, name: str) -> Union[str, None]:
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
    
    def set_as_axis(self, axis: str, name: Union[str, None], swap: bool = False) -> None:
        """Set column as a named axis or remove axis assignment if name is None.
        
        Args:
            axis: Name of the axis ('x', 'y', 'z', etc.)
            name: Column name, label, or None to remove assignment
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
        self.set_as_axis('x', column, swap)
        
    def set_as_y(self, column: Union[str, None], swap: bool = False) -> None:
        """Set column as y axis"""
        self.set_as_axis('y', column, swap)
    
    def set_as_z(self, column: Union[str, None], swap: bool = False) -> None:
        """Set column as z axis"""
        self.set_as_axis('z', column, swap)
    
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
    
    # Protected methods
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

        return all_translations.get(unit, unit)
    
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
            
    def _append_unit(self, column: str,
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
        formatted_unit = format(self.get_unit(column), use_format)
        return f'{column} {left}{formatted_unit}{right}'
    
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
            series = df[col]
            if hasattr(series.dtype, 'units'):
                unit = series.dtype.units
                df[col] = (series.pint.magnitude
                        .map(fmt.format)
                        .astype(f'pint[{unit}]'))
            else:
                df[col] = df[col].map(fmt.format)
    
    @update_column_names
    def _patched_rename(self, *args, **kwargs):
        """Handle column rename events"""
        return pd.DataFrame.rename

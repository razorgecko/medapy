from copy import deepcopy
from typing import Callable, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from medapy.utils import misc


@pd.api.extensions.register_dataframe_accessor("proc")
class DataProcessingAccessor():
    
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
    
    @staticmethod
    def _validate(obj):
        if '_ms_axes' not in obj.attrs:
            raise AttributeError("MeasurementSheet must be initialized")
        if len(obj.columns) < 2:
            raise ValueError("Data should have at least two columns")
    
    @property
    def ms(self) -> 'pd.DataFrame.MeasurementSheetAccessor':
        return self._obj.ms
    
    @property
    def x(self) -> pd.Series:
        return self.ms.x
    
    @property
    def y(self) -> pd.Series:
        return self.ms.y
    
    @property
    def z(self) -> pd.Series:
        return self.ms.y
    
    @property
    def col_x(self) -> pd.Series:
        return self.ms.axes['x']
    
    @property
    def col_y(self) -> pd.Series:
        return self.ms.axes['y']
    
    @property
    def col_z(self) -> pd.Series:
        return self.ms.axes['z']
    
    def check_monotonic(self, interrupt=False):
        # check = misc.check_monotonic_df(self._obj, self.col_x, interrupt=False)
        check = misc.check_monotonic_arr(self.x, interrupt=False)
        if interrupt and check == 0:
            raise ValueError(f'Column `{self.col_x}` is not monotonic')
        return check
            
    def ensure_increasing(self, inplace=False):
        # Work on a copy of the data
        df = self._get_df_copy()
        
        check = misc.check_monotonic_df(df, self.col_x, interrupt=False)
        if check == 0:
            raise ValueError(f'Column `{self.col_x}` is not monotonic')
        elif check == -1:
            # Invert the order in-place
            df.index = range(len(df) -1, -1, -1)
            df.sort_index(inplace=True)
            df.reset_index(drop=True, inplace=True)
            
        return self._if_inplace(df, inplace)
    
    def select_range(self,
                     val_range: npt.ArrayLike,
                     inside_range: bool = True,
                     inclusive: str = 'both',
                     handle_na: str = 'raise',
                     inplace: bool = False
                     ) -> pd.DataFrame | None:
        # Work on a copy of the data
        df = self._get_df_copy()
        # Get DataFrame with selected range
        result = misc.select_range_df(df, self.col_x, val_range, inside_range, inclusive, handle_na)
        # Alternative approach
        # df.drop(index=df.index.difference(result.index), inplace=True)
        result.reset_index(drop=True, inplace=True)
        
        return self._if_inplace(result, inplace)
    
    def filter_range(self,
                     val_range: npt.ArrayLike,
                     inside_range: bool = True,
                     inclusive: str = 'both',
                     handle_na: str = 'raise',
                     inplace: bool = False
                     ) -> pd.DataFrame | None:
        # Work on a copy of the data
        df = self._get_df_copy()
        # Get DataFrame with filtered range
        result = misc.filter_range_df(df, self.col_x, val_range, inside_range, inclusive, handle_na)
        # Alternative approach
        # df.drop(index=df.index.difference(result.index), inplace=True)
        result.reset_index(drop=True, inplace=True)
        
        return self._if_inplace(result, inplace)
    
    def symmetrize(self,
                cols: list[str] | str | None = None,
                *,
                append: str = '',
                set_axes: str | list[str] | None = None,
                add_labels: str | list[str] | None = None,
                inplace: bool = False
                ) -> pd.DataFrame | None:
        """
        Symmetrize values in specified columns and optionally create new columns.

        Parameters
        ----------
        cols : list[str] | str | None
            Column(s) to symmetrize. If None, y axis column is used.
        append : str
            String to append to column names for new columns. If empty, overwrites original columns.
        set_axes : str | list[str] | None
            Axis or axes to set for the new columns.
        add_labels : str | list[str] | None
            Label(s) to add to the new columns.
        inplace : bool
            If True, modify the DataFrame in place and return None.

        Returns
        -------
        pd.DataFrame or None
            Modified DataFrame or None if inplace=True.
            
        Notes
        -----
        See documentation for misc.symmetrize.
        """
        # Default to y axis column if None provided
        cols = self._prepare_values_list(cols, default=self.col_y, func=self.ms.get_column)
        n_cols = len(cols)

        # Prepare other parameters
        # keep existing units
        units = self._prepare_values_list(cols, default='', func=self.ms.get_unit, n=n_cols)
        set_axes = self._prepare_values_list(set_axes, default=None, n=n_cols)
        add_labels = self._prepare_values_list(add_labels, default=None, n=n_cols)

        # Generate new column names
        new_cols = [self._col_name_append(col, append) for col in cols]

        # Work on a copy of the data
        df = self._get_df_copy() 
        
        # Calculate symmetrized values
        new_values = [misc.symmetrize(df.ms[col]) for col in cols]

        # Assign values and metadata
        df.ms._set_column_states(columns=new_cols,
                                 values=new_values,
                                 units=units,
                                 axes=set_axes,
                                 labels=add_labels)

        return self._if_inplace(df, inplace)
    
    def antisymmetrize(self,
                cols: list[str] | str | None = None,
                *,
                append: str = '',
                set_axes: str | list[str] | None = None,
                add_labels: str | list[str] | None = None,
                inplace: bool = False
                ) -> pd.DataFrame | None:
        """
        Antisymmetrize values in specified columns and optionally create new columns.

        Parameters
        ----------
        cols : list[str] | str | None
            Column(s) to antisymmetrize. If None, y axis column is used.
        append : str
            String to append to column names for new columns. If empty, overwrites original columns.
        set_axes : str | list[str] | None
            Axis or axes to set for the new columns.
        add_labels : str | list[str] | None
            Label(s) to add to the new columns.
        inplace : bool
            If True, modify the DataFrame in place and return None.

        Returns
        -------
        pd.DataFrame or None
            Modified DataFrame or None if inplace=True.
            
        Notes
        -----
        See documentation for misc.antisymmetrize.
        """
        # Default to y axis column if None provided
        cols = self._prepare_values_list(cols, default=self.col_y, func=self.ms.get_column)
        n_cols = len(cols)

        # Prepare other parameters
        # keep existing units
        units = self._prepare_values_list(cols, default='', func=self.ms.get_unit, n=n_cols)
        set_axes = self._prepare_values_list(set_axes, default=None, n=n_cols)
        add_labels = self._prepare_values_list(add_labels, default=None, n=n_cols)

        # Generate new column names
        new_cols = [self._col_name_append(col, append) for col in cols]

        # Work on a copy of the data
        df = self._get_df_copy()
        
        # Calculate antisymmetrized values
        new_values = [misc.antisymmetrize(df.ms[col]) for col in cols]

        # Assign values and metadata
        df.ms._set_column_states(columns=new_cols,
                                 values=new_values,
                                 units=units,
                                 axes=set_axes,
                                 labels=add_labels)

        return self._if_inplace(df, inplace)
    
    def normalize(self,
                cols: list[str] | str | None = None,
                *,
                by: float | str,
                append: str = '',
                set_axes: str | list[str] | None = None,
                add_labels: str | list[str] | None = None,
                inplace: bool = False
                ) -> pd.DataFrame | None:
        """
        Normalize values in specified columns and optionally create new columns.

        Parameters
        ----------
        cols : list[str] | str | None
            Column(s) to normalize. If None, y axis column is used.
        by : float | str
            Value to use for normalization. Accepts 'first', 'mid', or 'last' as string value
        append : str
            String to append to column names for new columns. If empty, overwrites original columns.
        set_axes : str | list[str] | None
            Axis or axes to set for the new columns.
        add_labels : str | list[str] | None
            Label(s) to add to the new columns.
        inplace : bool
            If True, modify the DataFrame in place and return None.

        Returns
        -------
        pd.DataFrame or None
            Modified DataFrame or None if inplace=True.
            
        Notes
        -----
        Resulting values are assumed to be dimensionless.
        See documentation for misc.normalize.
        """
        # Default to y axis column if None provided
        cols = self._prepare_values_list(cols, default=self.col_y, func=self.ms.get_column)
        n_cols = len(cols)
        
        # Prepare other parameters
        units = self._prepare_values_list(None, default='', n=n_cols) # make dimensionless
        set_axes = self._prepare_values_list(set_axes, default=None, n=n_cols)
        add_labels = self._prepare_values_list(add_labels, default=None, n=n_cols)

        # Generate new column names
        new_cols = [self._col_name_append(col, append) for col in cols]

        # Work on a copy of the data
        df = self._get_df_copy()
        
        # Calculate normalized values
        new_values = [misc.normalize(df.ms[col], by=by) for col in cols]

        # Assign values and metadata
        df.ms._set_column_states(columns=new_cols,
                                 values=new_values,
                                 units=units,
                                 axes=set_axes,
                                 labels=add_labels)

        return self._if_inplace(df, inplace)
    
    def interpolate(
        self,
        x_new: npt.ArrayLike,
        cols: str | list[str] | None = None,
        *,
        interp: Callable[[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], npt.ArrayLike] | None = None,
        smooth: Callable[[npt.ArrayLike], npt.ArrayLike] | None = None,
        handle_na: str = 'raise',
        inplace: bool = False
        ) -> pd.DataFrame | None:
        """
        Interpolate and optionally smooth data for specific columns.

        Parameters
        ----------
        x_new : array_like
            Target x coordinates for interpolation.
        cols : list[str] | str | None
            Column(s) to interpolate. If None, y axis column is used.
        interp : callable, optional
            Custom interpolation function with signature f(x, y, x_new) -> y_new.
            If None, uses numpy.interp for linear interpolation.
        smooth : callable, optional
            Smoothing function with signature f(y) -> y_smooth.
            If None, no smoothing is applied.
        handle_na : str, default 'raise'
            How to handle NaN/Inf values: 'exclude' or 'raise'.
            If 'exclude', NaN/Inf excluded before range selecting. Returned data will not contain them.
            Pre-filtering is recommended for more complex NA handling strategies
        inplace : bool
            If True, modify the DataFrame in place and return None.

        Returns
        -------
        pd.DataFrame or None
            Modified DataFrame or None if inplace=True.
            
        Notes
        -----
        See documentation for misc.interpolate.
        """
        # Default to y axis column if None provided
        cols = self._prepare_values_list(cols, default=self.col_y, func=self.ms.get_column)
        
        # Create new df of only chosen columns
        df = self.ms[[self.col_x] + cols]

        # Calculate interpolated values
        new_values = [
            misc.interpolate(
                x=df.ms.x,
                y=df.ms[col],
                x_new=x_new,
                interp=interp,
                smooth=smooth,
                handle_na=handle_na)
            for col in cols]
        
        # Adjust the size of the column to x values
        df = df.reindex(np.arange(len(x_new)))
        df.ms[self.col_x] = x_new
        
        # Assign values and metadata
        df.ms._set_column_states(columns=cols, values=new_values)
        
        return self._if_inplace(df, inplace)
    
    def savgol_filter(
        self,
        window: int,
        cols: str | list[str] | None = None,
        *,
        order: int = 2,
        edge_mode: str | Callable = 'fill_after',
        fill_values: float | list[float, float] = np.nan,
        append: str = '',
        set_axes: str | list[str] | None = None,
        add_labels: str | list[str] | None = None,
        inplace: bool = False,
        **kwargs
        ) -> pd.DataFrame | None:
        """
        Apply Savitzky-Golay filter for specified columns.
        Optionally create new columns with filtered values.

        Parameters
        ----------
        window : int
            The length of the filter window.
            Correspond to window_length of scipy.signal.savgol_filter
        cols : list[str] | str | None
            Column(s) to smooth. If None, y axis column is used.
        order : int, default 2
            The order of the polynomial used to fit the samples; must be less than window.
            Correspond to polyorder of scipy.signal.savgol_filter
        edge_mode : str, default 'constant'
            Method to handle edges: see scipy.signal.savgol_filter.
            Additional option 'drop' is available to drop edge values after filtering
        fill_values : float or list, default np.nan
            Values for filling edges when appropriate
        append : str
            String to append to column names for new columns. If empty, overwrites original columns.
        set_axes : str | list[str] | None
            Axis or axes to set for the new columns.
        add_labels : str | list[str] | None
            Label(s) to add to the new columns.
        inplace : bool
            If True, modify the DataFrame in place and return None.
        **kwargs : 
            Additional arguments for misc.savgol_filter

        Returns
        -------
        pd.DataFrame or None
            Modified DataFrame or None if inplace=True.

        Notes
        -----
        See documentation for misc.savgol_filter.
        """
        # Default to y axis column if None provided
        cols = self._prepare_values_list(cols, default=self.col_y, func=self.ms.get_column)
        n_cols = len(cols)

        # Prepare other parameters
        # keep existing units
        units = self._prepare_values_list(cols, default='', func=self.ms.get_unit, n=n_cols)
        set_axes = self._prepare_values_list(set_axes, default=None, n=n_cols)
        add_labels = self._prepare_values_list(add_labels, default=None, n=n_cols)

        # Generate new column names
        new_cols = [self._col_name_append(col, append) for col in cols]

        
        # Create full dictionary of kwargs for misc.moving_average
        kwargs.update({'window': window, 'order': order,
                       'edge_mode': edge_mode, 'fill_values': fill_values})
        
        # Work on a copy of the data
        df = self._get_df_copy()
        
        # Calculate moving average values
        new_values = [misc.savgol_filter(df.ms[col], **kwargs) for col in cols]

        if edge_mode == 'drop':
            half_window = window // 2
            df = df.iloc[half_window:-half_window].reset_index(drop=True)
        
        # Assign values and metadata
        df.ms._set_column_states(columns=new_cols,
                                 values=new_values,
                                 units=units,
                                 axes=set_axes,
                                 labels=add_labels)

        return self._if_inplace(df, inplace)
    
    def moving_average(
        self,
        window: int, 
        cols: str | list[str] | None = None,
        *,
        edge_mode: str | Callable = 'fill_after',
        fill_values: float | list[float, float] = np.nan,
        append: str = '',
        set_axes: str | list[str] | None = None,
        add_labels: str | list[str] | None = None,
        inplace: bool = False,
        **kwargs
        ) -> pd.DataFrame | None:
        """
        Calculate moving average with a centered window for specified columns.
        Optionally create new columns with smoothed values.

        Parameters
        ----------
        window : int
            Size of the moving average window
        cols : list[str] | str | None
            Column(s) to smooth. If None, y axis column is used.
        edge_mode : str or callable
            Method to handle edges: 'fill_after', 'drop', 'repeat', or any numpy.pad mode
        fill_values : float or list, default np.nan
            Values for filling edges when appropriate
        append : str
            String to append to column names for new columns. If empty, overwrites original columns.
        set_axes : str | list[str] | None
            Axis or axes to set for the new columns.
        add_labels : str | list[str] | None
            Label(s) to add to the new columns.
        inplace : bool
            If True, modify the DataFrame in place and return None.
        **kwargs : 
            Additional arguments for misc.moving_average

        Returns
        -------
        pd.DataFrame or None
            Modified DataFrame or None if inplace=True.

        Notes
        -----
        See documentation for misc.moving_average.
        """
        # Default to y axis column if None provided
        cols = self._prepare_values_list(cols, default=self.col_y, func=self.ms.get_column)
        n_cols = len(cols)

        # Prepare other parameters
        # keep existing units
        units = self._prepare_values_list(cols, default='', func=self.ms.get_unit, n=n_cols)
        set_axes = self._prepare_values_list(set_axes, default=None, n=n_cols)
        add_labels = self._prepare_values_list(add_labels, default=None, n=n_cols)

        # Generate new column names
        new_cols = [self._col_name_append(col, append) for col in cols]

        # Create full dictionary of kwargs for misc.moving_average
        kwargs.update({'window': window, 'edge_mode': edge_mode, 'fill_values': fill_values})
        
        # Work on a copy of the data
        df = self._get_df_copy()
        
        # Calculate moving average values
        new_values = [misc.moving_average(df.ms[col], **kwargs) for col in cols]

        if edge_mode == 'drop':
            half_window = window // 2
            df = df.iloc[half_window:-half_window].reset_index(drop=True)
        
        # Assign values and metadata
        df.ms._set_column_states(columns=new_cols,
                                 values=new_values,
                                 units=units,
                                 axes=set_axes,
                                 labels=add_labels)

        return self._if_inplace(df, inplace)


    # Protected methods
    def _get_df_copy(self) -> pd.DataFrame:
        return self._obj.copy(deep=True) # default deep=True, but to be sure
    
    def _if_inplace(self, result, inplace: bool) -> pd.DataFrame | None:
        # For inplace, explicitly update both data and attrs
        if inplace:
            # Update data using pandas NDFrame method _update_inplace (pandas/core/generic.py)
            # This allows to assign df._mgr directly avoiding any
            # discrepancies between data dtypes or length of dataframes
            self._obj._update_inplace(result)  
            self._obj.attrs = deepcopy(result.attrs) # Explicitly copy modified attrs
            return None
        else:
            return result  # Return modified result without calling __finalize__
           
    @staticmethod
    def _prepare_values_list(values: str | list | None,
                            default: Any = None,
                            n: int = 1,
                            func: Callable | None = None,
                            ) -> list:
        """
        Prepare a consistent list of values with specified length.

        Parameters
        ----------
        values : str, list, or None
            Input values to prepare as a list.
        default : Any, default None
            Default value to use if values is None.
        n : int, default 1
            Expected length of the resulting list.
        func : Callable, optional
            Function to apply to resulting list elements if provided.

        Returns
        -------
        list
            A list of length n containing the values or defaults.

        Notes
        -----
        - If values is None, returns a list of n defaults
        - If values is a string or a non-iterable, returns a single-item list
        - If values is a list shorter than n, raises ValueError
        - If values is a list of correct length, returns it unchanged
        """
        if func is not None and not callable(func):
            raise TypeError("'func' should be Callable or None")
            
        if values is None:
            return [default] * n
        
        if isinstance(values, list) and n == 1:
            if func is not None:
                return [func(val) for val in values]
            return values
        
        # Handle non-iterable values (including strings) as single items
        if isinstance(values, str) or not hasattr(values, '__iter__'):
            values = [values]
        # Convert other non-list iterables to list
        elif not isinstance(values, list):
            values = list(values)

        # Ensure values is a list by this point
        if not isinstance(values, list):
            raise TypeError(f"Expected string, list, or None, got {type(values).__name__}")

        # Check if list length matches expected length
        if len(values) == 1 and n > 1:
            # Special case: if we have a single value but need n values,
            # repeat the single value n times
            values = values * n
        elif len(values) != n:
            raise ValueError(f"Expected list of length {n}, got length {len(values)}")

        if func is not None:
            return [func(val) for val in values]
        return values
        
    @staticmethod
    def _col_name_append(column, append, sep='_'):
        if append:
            return f"{column}{sep}{append}"
        return column
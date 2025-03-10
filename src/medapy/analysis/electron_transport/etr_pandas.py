from typing import Callable, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pint

import medapy.utils.misc as misc
from medapy.analysis.proc_pandas import DataProcessingAccessor
from . import electron_transport as etr


ureg = pint.get_application_registry()

@pd.api.extensions.register_dataframe_accessor("etr")
class ElectricalTransportAccessor(DataProcessingAccessor):
    def r2rho(self,
              kind: str,
              col: str | None = None,
              *,
              t: float,
              width: float = None,
              length: float = None,
              new_col: str = 'rho',
              set_axis: str | None = None,
              add_label : str | None = None,
              inplace: bool = False
              ) -> pd.DataFrame | None:
        # If t, width, and length are floats, it is assumed they are in meter units
        
        # Default to y axis column if None provided
        col = self.col_y if col is None else self.ms.get_column(col)
        
        if hasattr(t, 'units'):
            t = t.to('m')
            t, t_unit = t.magnitude, t.units
        else:
            t_unit = ureg.Unit('m')
            
        if hasattr(width, 'units') and hasattr(length, 'units'):
            width = width.to('m').magnitude
            length = length.to('m').magnitude
        elif hasattr(width, 'units') ^ hasattr(length, 'units'): # XOR
            # only one of them has units
            raise AttributeError("Only one of width and length have units")
        
        # Work on a copy of the data
        df = self._get_df_copy()
        
        # Should we convert r to 'ohm' before calculating 'rho'?
        r_unit = pint.Unit(df.ms.get_unit(col))
        unit = r_unit * t_unit
        # new_col = f"{new_col}_{kind}"
        
        # Calculate resistivity values
        new_values = etr.r2rho(df.ms[col], kind=kind, t=t, width=width, length=length)
        
        # Assign values and metadata
        df.ms._set_column(new_col, new_values, unit, set_axis, add_label)

        return self._if_inplace(df, inplace)
        
    def fit_linhall(self,
                    col: str | None = None,
                    x_range: npt.ArrayLike | None = None,
                    *,
                    add_col: str = 'linHall',
                    set_axis: str | None = None,
                    add_label: str | None = None,
                    inplace: bool = False
                    ) -> tuple[np.ndarray, pd.DataFrame | None]:
        # Default to y axis column if None provided
        col = self.col_y if col is None else self.ms.get_column(col)
        
        # Calculate fit coefficients
        coefs = misc.quick_fit(self.x, self.ms[col], x_range=x_range)
        
        # Work on a copy of the data
        df = self._get_df_copy()
        
        if add_col:
            # Prepare metadata
            unit = df.ms.get_unit(col)
            # Prepare new column name
            new_col = self._col_name_append(col, append=add_col)
            # Calculate fit values
            new_values =  misc.make_curve(df.ms.x, coefs)
            # Assign values and metadata
            df.ms._set_column(new_col, new_values, unit, set_axis, add_label)

        return coefs, self._if_inplace(df, inplace)
        
    def fit_twoband(self,
                    p0: tuple[float, float, float, float],
                    col: str | None = None,
                    *,
                    kind: str,
                    bands: str,
                    field_range: npt.ArrayLike | None = None,
                    inside_range: bool = True,
                    extension: tuple | npt.ArrayLike | pd.DataFrame | None = None,
                    add_col: str | None = '2bnd',
                    set_axis: str | None = None, add_label : str | None = None,
                    inplace: bool = False,
                    **kwargs) -> tuple[tuple, pd.DataFrame | None]:
        # Default to y axis column if None provided
        col = self.col_y if col is None else self.ms.get_column(col)
        
        if isinstance(extension, pd.DataFrame):
            if '_ms_axes' in extension.attrs:
                extension = (extension.ms.x, extension.ms.y)
            else:
                extension = (extension.iloc[:, 0], extension.iloc[:, 1])
        
        # Work with particular columns      
        field, rho = self.x, self.ms[col]
        if field_range:
            fldrho = np.column_stack((self.x, self.ms[col]))   
            fldrho = misc.select_range_arr(fldrho, 0, field_range, inside_range=inside_range)
            field, rho = fldrho.T
        
        # Calculate fit coefficient        
        coefs = etr.fit_twoband(field, rho, p0, kind=kind, bands=bands, extension=extension, **kwargs)
        
        # Work on a copy of the data
        df = self._get_df_copy()
        
        if add_col:
            if kind == 'xx':
                func_2bnd = etr.gen_mr2bnd_eq(bands)
            else:
                func_2bnd = etr.gen_hall2bnd_eq(bands)
            # Prepare metadata
            unit = df.ms.get_unit(col)
            # Prepare new column name
            new_col = self._col_name_append(col, append=f'{add_col}{bands}')
            # Calculate fit values
            new_values = func_2bnd(df.ms.x, *coefs)
            # Assign values and metadata
            df.ms._set_column(new_col, new_values, unit, set_axis, add_label)

        return coefs, self._if_inplace(df, inplace)
    
    def calculate_twoband(self,
                          p: tuple[float, float, float, float],
                          cols: str | list[str] | None = None,
                          *,
                          kinds: str | list[str],
                          bands: str,
                          append: str = '2bnd',
                          set_axes: str | list[str] | None = None,
                          add_labels: str | list[str] | None = None,
                          inplace: bool = False
                          ) -> pd.DataFrame | None:
        # Default to y axis column if None provided
        cols = self._prepare_values_list(cols, default=self.col_y, func=self.ms.get_column)
        n_cols = len(cols)

        # Prepare other parameters
        # keep existing units
        units = self._prepare_values_list(cols, default='', func=self.ms.get_unit, n=n_cols)
        set_axes = self._prepare_values_list(set_axes, default=None, n=n_cols)
        add_labels = self._prepare_values_list(add_labels, default=None, n=n_cols)
        appendices = self._prepare_values_list(append, default='2bnd', n=n_cols, func=lambda x: x + bands)
        
        # Generate new column names
        new_cols = misc.apply(self._col_name_append, column=cols, append=appendices)
        
        # Prepare twoband functions list
        def kind2func(kind, bands):
            mapping = {'xx': etr.gen_mr2bnd_eq(bands),
                       'xy': etr.gen_hall2bnd_eq(bands)}
            return mapping.get(kind)
        
        kinds = self._prepare_values_list(kinds, default=None, n=n_cols)
        funcs = self._prepare_values_list(kinds, default=None,
                                          func=lambda x: kind2func(x, bands),
                                          n=n_cols)
        
        # Work on a copy of the data
        df = self._get_df_copy()
        
        # Calculate fit values
        new_values = [func(self.x, *p) for func in funcs]

        # Assign values and metadata
        misc.apply(df.ms._set_column,
                column=new_cols,
                values=new_values,
                unit=units,
                axis=set_axes,
                label=add_labels)
        
        return self._if_inplace(df, inplace)


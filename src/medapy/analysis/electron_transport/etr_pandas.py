from typing import Callable

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
        new_col = f"{new_col}_{kind}"
        
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
                    ) -> np.ndarray | list[np.ndarray]:
        
        
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
                    **kwargs) -> tuple[float, float, float, float]:
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
            new_col = self._col_name_append(col, append=add_col)
            # Calculate fit values
            new_values = func_2bnd(df.ms.x, *coefs)
            # Assign values and metadata
            df.ms._set_column(new_col, new_values, unit, set_axis, add_label)

        return coefs, self._if_inplace(df, inplace)
        
    # @staticmethod
    # def convert_twoband_params_to_cm(params):
    #     return np.asarray(params) * np.array([1e-6, 1e-6, 1e4, 1e4])
    
    
    # @classmethod
    # def __get_twoband_string_res(cls, params, bands, rho_coef=None):
    #     p = cls.convert_twoband_params_to_cm(params)
    #     str_res = f'n_1 = {p[0]:.2E} cm^-3\nn_2 = {p[1]:.2E} cm^-3'
    #     str_res += f'\nmu_1 = {p[2]:.2f} cm^2/V/s\nmu_2 = {p[3]:.2f} cm^2/V/s'
    #     twoband_xx = cls.generate_twoband_eq(kind='xx', bands='he')
    #     rho_xx0 = twoband_xx(0, *params)
    #     if rho_coef is not None:
    #         str_res += f'\nRxx(H=0) = {rho_xx0/rho_coef:.2f} Ohms'
    #     else:
    #         str_res += f'\nrho_xx(H=0) = {rho_xx0:.2E} Ohms*m'
    #     return str_res
    
    # @classmethod
    # def __get_linear_string_res(cls, params, t=None):
    #     str_res = f'a0 = {params[0]:.2f} Ohms'
    #     str_res += f'\nk = {params[1]:.2E} Ohms/T'
    #     if t is not None:
    #         n = 1e-6/(params[1]*e*t)
    #         str_res += f'\nn = {n:.2E} cm^-3'
    #     return str_res
    
    # @classmethod        
    # def params_to_str(cls, params, *, kind='twoband', bands='he', W=None, L=None, t=None):
    #     match kind:
    #         case 'linear':
    #             return cls.__get_linear_string_res(params, t)
    #         case 'twoband':
    #             if W is not None and L is not None and t is not None:
    #                 rho_coef = W*t/L
    #             else:
    #                 rho_coef = None
    #             return cls.__get_twoband_string_res(params, bands, rho_coef)          
    
    
    
    
    
    

    
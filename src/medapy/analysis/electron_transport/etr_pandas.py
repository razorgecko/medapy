from pathlib import Path
from typing import Callable, Optional, TextIO, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from pint_pandas import PintType

import medapy.utils.misc as misc
from . import electron_transport as etr


ureg = PintType.ureg

@pd.api.extensions.register_dataframe_accessor("etr")
class ElectricalTransportAccessor():
    
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
    def x(self) -> pd.Series:
        return self._obj.ms.x
    
    @property
    def y(self) -> pd.Series:
        return self._obj.ms.y
    
    @property
    def z(self) -> pd.Series:
        return self._obj.ms.y
    
    @property
    def _col_x(self) -> pd.Series:
        return self._obj.ms.axes['x']
    
    @property
    def _col_y(self) -> pd.Series:
        return self._obj.ms.axes['y']
    
    @property
    def _col_z(self) -> pd.Series:
        return self._obj.ms.axes['z']
    
    def check_monotonic(self, interrupt=False):
        # check = misc.check_monotonic_df(self._obj, self._col_x, interrupt=False)
        check = misc.check_monotonic_arr(self.x, col=0, interrupt=False)
        if interrupt and check == 0:
            raise ValueError(f'Column `{self._col_x}` is not monotonic')
        return check
            
    def ensure_increasing(self, inplace=False):
        # check = misc.check_monotonic_df(self._obj, self._col_x, interrupt=False)
        df = self.__if_inplace(inplace)
        check = misc.check_monotonic_arr(self.x, col=0, interrupt=False)
        if check == 0:
            raise ValueError(f'Column `{self._col_x}` is not monotonic')
        elif check == -1:
            # Invert the order in-place
            df.index = range(len(df) -1, -1, -1)
            df.sort_index(inplace=True)
            df.reset_index(drop=True, inplace=True)
        if not inplace:
            return df
    
    def select_range(self, val_range: npt.ArrayLike,
        inside_range: bool = True, inclusive: str = 'both', handle_na: str = 'raise',
        inplace: bool = False
        ) -> None:
        df = self.__if_inplace(inplace)
        result = misc.select_range_df(df, self._col_x, val_range, inside_range, inclusive, handle_na)
        df.drop(index=df.index.difference(result.index), inplace=True)
        if not inplace:
            return df
    
    def symmetrize(self, add_col='sym', inplace=False):
        df = self.__if_inplace(inplace)
        y_new = misc.symmetrize(self.y)
        if add_col:
            col_sym = f"{self._col_y} {add_col}"
            unit = df.ms.get_unit(self._col_y)
            df[col_sym] = pd.Series(y_new, dtype=f"pint[{unit}]")
        else:
            df.ms.replace_values(self._col_y, y_new)
        if not inplace:
            return df
                
    def antisymmetrize(self, add_col='antisym', inplace=False):
        df = self.__if_inplace(inplace)
        y_new = misc.antisymmetrize(self.y)
        if add_col:
            col_sym = f"{self._col_y} {add_col}"
            unit = df.ms.get_unit(self._col_y)
            df[col_sym] = pd.Series(y_new, dtype=f"pint[{unit}]")
        else:
            df.ms.replace_values(self._col_y, y_new)
        if not inplace:
            return df
    
    def normalize(self, by=None, add_col='norm', inplace=False):
        #inplace=False is not implemented
        df = self.__if_inplace(inplace)
        y_new = misc.normalize(self.y, by)
        if add_col:
            col_norm = f"{self._col_y} {add_col}"
            df[col_norm] = pd.Series(y_new, dtype="pint[]")
        else:
            df[self._col_y] = pd.Series(y_new, dtype="pint[]")
        if not inplace:
            return df
    
    def interpolate(self,
        x_new: npt.ArrayLike,
        *,
        interp_method: Callable[[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], npt.ArrayLike] | None = None,
        smooth_method: Callable[[npt.ArrayLike], npt.ArrayLike] | None = None,
        handle_na: str = 'raise'
        ) -> pd.DataFrame:

        y_new = misc.interpolate(self.x, self.y, x_new,
                                 interp_method=interp_method, smooth_method=smooth_method, handle_na=handle_na)
        
        return self._form_new_xy_df(x_new, y_new)
    
    def r2rho(self, kind: str,
              t: float, width: float = None, length: float = None,
              add_col: str = 'rho',
              set_axis: str | None = None, add_label : str | None = None,
              inplace: bool = False) -> pd.DataFrame | pd.Series | None:
        # If t, width, and length are floats, it is assumed they are in meter units
        df = self.__if_inplace(inplace)
        if hasattr(t, 'units'):
            t = t.to('m')
            t, t_unit = t.magnitude, t.units
        else:
            t_unit = ureg.Unit('m')
        if hasattr(width, 'units') and hasattr(length, 'units'):
            width = width.to('m')
            width, width_unit = width.magnitude, width.units
            length = length.to('m')
            length, length_unit = length.magnitude, length.units
        elif (not hasattr(width, 'units')) and (not hasattr(length, 'units')):
            width_unit, length_unit = ureg.Unit('m'), ureg.Unit('m')
        else: # only one of them has units
            raise AttributeError("Only one of width and length have different units")
        # Should we convert r to 'ohm' before calculating 'rho'?
        r_unit = df.ms.get_unit(self._col_y)
        
        rho = etr.r2rho(self.y, kind=kind, t=t, width=width, length=length)
        rho_unit = r_unit * t_unit
        rho_series = pd.Series(rho, dtype=f"pint[{rho_unit}]")
        if add_col:
            col_rho = f"{add_col}_{kind}"
            df[col_rho] = rho_series
            self.__setax_addlbl(col_rho, set_axis, add_label)
        else:
            return rho_series
        if not inplace:
            return df
        
    
    def fit_linhall(self, x_range, add_col='linHall', a0_col=None, set_axis=None, add_label=None):
        df = self._obj
        coefs = misc.quick_fit(self.x, self.y, x_range=x_range)
        if add_col:
            if a0_col is not None:
                use_coefs = np.array([a0_col, coefs[1]]) 
            else:
                use_coefs = coefs
            col_fit = f"{self._col_y}_{add_col}"
            unit = df.ms.get_unit(self._col_y)
            series = misc.make_curve(self.x, use_coefs)
            df[col_fit] = pd.Series(series, dtype=f"pint[{unit}]")
            self.__setax_addlbl(col_fit, set_axis, add_label)
        return coefs
    
    def fit_twoband(self, p0: tuple[float, float, float, float], 
                    *, kind: str, bands: str,
                    field_range: npt.ArrayLike | None = None,
                    inside_range: bool = True,
                    extension: tuple | npt.ArrayLike | pd.DataFrame | None = None,
                    add_col: str | None = '2bnd',
                    set_axis: str | None = None, add_label : str | None = None,
                    **kwargs) -> tuple[float, float, float, float]:
        df = self._obj
        if isinstance(extension, pd.DataFrame):
            if '_ms_axes' in extension.attrs:
                extension = (extension.ms.x, extension.ms.y)
            else:
                extension = (extension.iloc[:, 0], extension.iloc[:, 1])
        if field_range:
            fldrho = np.column_stack((self.x, self.y))   
            fldrho = misc.select_range_arr(fldrho, 0, field_range, inside_range=inside_range)
            field, rho = fldrho.T
        else:
            field, rho = self.x, self.y
                
        coefs = etr.fit_twoband(field, rho, p0, kind=kind, bands=bands, extension=extension, **kwargs)
        
        if add_col:
            if kind == 'xx':
                func_2bnd = etr.gen_mr2bnd_eq(bands)
            else:
                func_2bnd = etr.gen_hall2bnd_eq(bands)
            col_fit = f"{self._col_y}_{add_col}_{bands}"
            df[col_fit] = pd.Series(func_2bnd(self.x, *coefs), dtype='pint[ohm*m]')
            self.__setax_addlbl(col_fit, set_axis, add_label)
        return coefs
    
    def __if_inplace(self, inplace: bool) -> pd.DataFrame:
        if inplace:
            return self._obj
        return self._obj.copy(deep=True)
    
    def __setax_addlbl(self, column: str, axis: str | None, label: str | None) -> None:
        df = self._obj
        if axis is not None:
            df.ms.set_as_axis(axis, column)
        if label is not None:
            df.ms.add_label(label, column)
    
    def _form_new_xy_df(self, x_new, y_new):
        df_new = pd.DataFrame({self._col_x: x_new, self._col_y: y_new})
        df_new.ms.init_msheet(units=False)
        unit_x = self._obj.ms.get_unit(self._col_x)
        unit_y = self._obj.ms.get_unit(self._col_y)
        df_new.ms.set_unit(self._col_x, unit_x)
        df_new.ms.set_unit(self._col_y, unit_y)
        return df_new
    
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
    
    
    
    
    
    

    
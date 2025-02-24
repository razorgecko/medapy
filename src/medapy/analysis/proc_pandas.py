from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

import medapy.utils.misc as misc


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
        # check = misc.check_monotonic_df(self._obj, self.col_x, interrupt=False)
        df = self.__if_inplace(inplace)
        check = misc.check_monotonic_df(df, self.col_x, interrupt=False)
        if check == 0:
            raise ValueError(f'Column `{self.col_x}` is not monotonic')
        elif check == -1:
            # Invert the order in-place
            df.index = range(len(df) -1, -1, -1)
            df.sort_index(inplace=True)
            df.reset_index(drop=True, inplace=True)
        if not inplace:
            return df
    
    def select_range(self,
                     val_range: npt.ArrayLike,
                     inside_range: bool = True,
                     inclusive: str = 'both',
                     handle_na: str = 'raise',
                     inplace: bool = False
                     ) -> pd.DataFrame | None:
        df = self.__if_inplace(inplace)
        result = misc.select_range_df(df, self.col_x, val_range, inside_range, inclusive, handle_na)
        df.drop(index=df.index.difference(result.index), inplace=True)
        df.reset_index(drop=True, inplace=True)
        if not inplace:
            return df
    
    def symmetrize(self,
                   cols: list[str] = [],
                   *,
                   add_col: str = '',
                   set_axes: str | list[str] | None = None,
                   add_labels : str | list[str] | None = None,
                   inplace: bool = False
                   ) -> pd.DataFrame | None:
        df = self.__if_inplace(inplace)
        func = misc.symmetrize
        if inplace:
            return self._apply_func_to_cols(df, func, cols=cols, add_col=add_col,
                                            set_axes=set_axes, add_labels=add_labels)
    
    def antisymmetrize(self,
                   cols: list[str] = [],
                   *,
                   add_col: str = '',
                   set_axes: str | list[str] | None = None,
                   add_labels : str | list[str] | None = None,
                   inplace: bool = False
                   ) -> pd.DataFrame | None:
        df = self.__if_inplace(inplace)
        func = misc.antisymmetrize
        if inplace:
            return self._apply_func_to_cols(df, func, cols=cols, add_col=add_col,
                                            set_axes=set_axes, add_labels=add_labels)

    def normalize(self,
                  by: str | np.number,
                  cols: list[str] = [],
                  *,
                  add_col: str = '',
                  set_axes: str | list[str] | None = None,
                  add_labels : str | list[str] | None = None,
                  inplace: bool = False
                  ) -> pd.DataFrame | None:
        df = self.__if_inplace(inplace)
        def norm(seq):
            return misc.normalize(seq, by=by)
        func = norm
        
        if inplace:
            return self._apply_func_to_cols(df, func, cols=cols, add_col=add_col,
                                            set_axes=set_axes, add_labels=add_labels)
    
    def interpolate(
        self,
        x_new: npt.ArrayLike,
        cols: str | list[str] = [],
        *,
        interp_method: Callable[[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], npt.ArrayLike] | None = None,
        smooth_method: Callable[[npt.ArrayLike], npt.ArrayLike] | None = None,
        handle_na: str = 'raise'
        ) -> pd.DataFrame:
        cols = self._prepare_values_list(cols, self.col_y)
        df = self.ms[[self.col_x] + cols]
        df = df.reindex(np.arange(len(x_new)))
        df.ms[self.col_x] = x_new
        
        def interp(seq):
            return misc.interpolate(self.x, seq, x_new,
                                    interp_method=interp_method,
                                    smooth_method=smooth_method,
                                    handle_na=handle_na)
        func = interp
        
        return self._apply_func_to_cols(df, func, cols=cols)
    
    def _apply_func_to_cols(self,
                            df: pd.DataFrame,
                            func: Callable[[npt.ArrayLike], npt.ArrayLike],
                            cols: str | list[str] = [],
                            *,
                            add_col='',
                            set_axes: str | list[str] | None = None,
                            add_labels : str | list[str] | None = None,
                            ) -> pd.DataFrame:
    
        cols, set_axes, add_labels = self._prepare_cols_axes_labels(cols, set_axes, add_labels)
        
        for i, column in enumerate(cols):
            col = self.ms.get_column(column)
            if add_col:
                col_name = f"{col}_{add_col}"
            else:
                col_name = col
            vals = func(self.ms[col])
            df.ms[col_name] = vals
            self.__setax_addlbl(col_name, set_axes[i], add_labels[i])
        return df
    
    def _prepare_cols_axes_labels(self, cols, axes, labels):

        cols = self._prepare_values_list(cols, self.col_y)
        n_cols = len(cols)
        
        axes = self._prepare_values_list(axes, None, n_cols)
        n_axes = len(axes)
        if n_axes != n_cols:
            raise ValueError(f"Number of axes ({n_axes}) must correspond "
                             f"to number of columns ({n_cols})")  
        
        labels = self._prepare_values_list(labels, None, n_cols)
        n_labels = len(labels)
        if n_labels != n_cols:
            raise ValueError(f"Number of labels ({n_labels}) must correspond "
                             f"to number of columns ({n_cols})")
        return cols, axes, labels
    
    @staticmethod
    def _prepare_values_list(values: str | list[str] | None,
                             default: str | None,
                             n: int = 1,
                             ) -> list[str | None]:
        if not values:
            values = [default] * n
        elif isinstance(values, str):
            values = [values]
        return values
        
    def __if_inplace(self, inplace: bool) -> pd.DataFrame:
        if inplace:
            return self._obj
        return self._obj.copy(deep=True)
    
    def __setax_addlbl(self, column: str, axis: str | None, label: str | None) -> None:
        df = self._obj
        if axis is not None:
            df.ms.set_as_axis(column, axis)
        if label is not None:
            df.ms.add_labels({column: label})
    
    def _form_new_xy_df(self, x_new, y_new):
        # df_new = pd.DataFrame({self.col_x: x_new, self.col_y: y_new})
        n = x_new.shape[0]
        df_new = self.ms[[self.col_x, self.col_y]]
        df_new = self._obj.reindex(np.arange(n))
        # df_new.ms.init_msheet(units=False)
        # unit_x = self._obj.ms.get_unit(self.col_x)
        # unit_y = self._obj.ms.get_unit(self.col_y)
        # df_new.ms.set_unit(self.col_x, unit_x)
        # df_new.ms.set_unit(self.col_y, unit_y)
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
    
    
    
    
    
    

    
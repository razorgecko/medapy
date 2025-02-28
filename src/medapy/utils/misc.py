import warnings
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd


# Python functionality
def apply(func, **kwargs):
    if not isinstance(func, Callable):
        raise TypeError("'func' value should be callable")
    first_param = True
    expected_length = None
    for param, values_list in kwargs.items():
        try:
            values_list[0]
            current_length = len(values_list)
            if first_param:
                expected_length = current_length
                first_param = False
            elif current_length != expected_length:
                raise ValueError(f"Inconsistent lengths: '{param}' has length {current_length}, "
                                    f"expected {expected_length}")
            
        except TypeError:
            raise TypeError(f"Subscriptable sequences are required for batch processing. "
                            f"The '{param}' value '{values_list}' can not be used.")
        except IndexError:
            raise IndexError(f"'{param}' value is empty")
    
    # results = []
    # for row in zip(*kwargs.values()):
    #     kw_dict = dict(zip(kwargs.keys(), row))
    #     results.append(func(**kw_dict))
    # return results

    keys = list(kwargs.keys())
    values = zip(*kwargs.values())

    return [func(**dict(zip(keys, row))) for row in values]

# Pandas functionality   
def check_monotonic_df(df: pd.DataFrame, col: str, interrupt: bool = False) -> int:
    """Check if a DataFrame column is monotonic.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the column to check
    col : str
        Name of the column to check for monotonicity
    interrupt : bool, optional
        If True, raises ValueError for non-monotonic data instead of returning 0,
        by default False

    Returns
    -------
    int
        Monotonicity indicator:
         1 : monotonically increasing
         0 : not monotonic
        -1 : monotonically decreasing

    Raises
    ------
    TypeError
        If inputs have incorrect types
    ValueError
        If column doesn't exist in DataFrame
        If column contains non-numeric data
        If column is empty
        If interrupt=True and the column is not monotonic

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3]})
    >>> check_data_monotonic(df, 'a')
    1
    >>> df = pd.DataFrame({'a': [3, 2, 1]})
    >>> check_data_monotonic(df, 'a')
    -1
    >>> df = pd.DataFrame({'a': [1, 3, 2]})
    >>> check_data_monotonic(df, 'a')
    0
    """
    # Type checks
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(col, str):
        raise TypeError("col must be a string")
    if not isinstance(interrupt, bool):
        raise TypeError("interrupt must be a boolean")

    # Check if column exists
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    x = df[col]

    # Check for empty column
    if len(x) == 0:
        raise ValueError(f"Column '{col}' is empty")

    # Check for non-numeric data
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError(f"Column '{col}' contains non-numeric data")

    # Check for NaN values
    if x.isna().any():
        raise ValueError(f"Column '{col}' contains NaN values")

    if x.is_monotonic_increasing:
        return 1
    elif x.is_monotonic_decreasing:
        return -1
    else:
        if interrupt:
            raise ValueError(f"Column `{col}` is not monotonic")
        return 0

def select_range_df(
    df: pd.DataFrame,
    col: str,
    val_range: npt.ArrayLike,
    inside_range: bool = True,
    inclusive: str = 'both',
    handle_na: str = 'raise'
    ) -> pd.DataFrame:
    """
    Select data rows based on value range using binary search.
    Only suitable for SORTED sequences - much faster than filter_range.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input data with sorted values in specified column
    col : str
        Column to select on (must be monotonically increasing/decreasing)
    val_range : tuple or list
        (min_value, max_value) defining the range
    inside_range : bool, default True
        If True, keep values within range
        If False, keep values outside range
    inclusive : str, default 'both'
        Include boundaries: 'both', 'neither', 'left', 'right'
    handle_na : str, default 'raise'
        How to handle NaN/inf values: 'exclude' or 'raise'

    Returns:
    --------
    pandas.DataFrame
        Selected data

    Raises:
    -------
    ValueError
        If the column is not monotonic or if parameters are invalid
    KeyError
        If data is DataFrame, and column not found
        
    Notes:
    ------
    Supports open boundaries using NaN values:
    - (NaN, x) means "less than or equal to x"
    - (x, NaN) means "greater than or equal to x"
    - (NaN, NaN) means "select all" if inside_range=True, "select none" if inside_range=False
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data must be a DataFrame object")
    if not isinstance(col, str):
        raise TypeError("col must be a string for DataFrame input")
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")
    if not isinstance(inside_range, bool):
        raise TypeError("inside_range must be a boolean")
    inclusive = _validate_option(inclusive, ['both', 'neither', 'left', 'right'], 'inclusive')
    handle_na = _validate_option(handle_na, ['exclude', 'raise'], 'handle_na')
    left, right = _validate_val_range(val_range)
    
    x = df[col].astype(float) # works x100 slower if dtype is pint
    # Handle NaN/inf values
    mask_na = x.isna() | x.isin([np.inf, -np.inf])
    if mask_na.any():
        if handle_na == 'raise':
            raise ValueError("NaN or infinite values found in data")
        elif handle_na == 'exclude':
            df = df[~mask_na]
    
    # Check if sequence is monotonic
    is_increasing = x.is_monotonic_increasing
    is_decreasing = x.is_monotonic_decreasing
    if not (is_increasing or is_decreasing):
        raise ValueError(f"Column '{col}' must be monotonically increasing or decreasing")

    left_idx, right_idx = _get_range_boundary_indices(x, left, right, inclusive, is_decreasing)

    mask = np.zeros(len(df), dtype=bool)
    mask[left_idx:right_idx] = True
    # Apply inside_range logic
    return df[mask if inside_range else ~mask]

def filter_range_df(
    df: pd.DataFrame,
    col: str,
    val_range: npt.ArrayLike,
    inside_range: bool = True,
    inclusive: str = 'both',
    handle_na: str = 'raise'
    ) -> pd.DataFrame:
    """
    Filter DataFrame rows based on whether values in specified column are within given range.
    Uses comparison operators - suitable for any sequence (sorted or not).

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    col : str
        Column name to filter on
    val_range : tuple or list
        (min_value, max_value) defining the range
    inside_range : bool, default True
        If True, keep values within range
        If False, keep values outside range
    inclusive : str, default 'both'
        Include boundaries: 'both', 'neither', 'left', 'right'
    handle_na : str, default 'raise'
        How to handle NaN/inf values: 'exclude' or 'raise'

    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame

    Raises:
    -------
    ValueError
        If parameters are invalid or if NaN/inf found with handle_na='raise'
    KeyError
        If column not found in DataFrame
    
    Notes:
    ------
    Supports open boundaries using NaN values:
    - (NaN, x) means "less than or equal to x"
    - (x, NaN) means "greater than or equal to x"
    - (NaN, NaN) means "select all" if inside_range=True, "select none" if inside_range=False
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")
    if not isinstance(inside_range, bool):
        raise TypeError("inside_range must be a boolean")
    inclusive = _validate_option(inclusive, ['both', 'neither', 'left', 'right'], 'inclusive')
    handle_na = _validate_option(handle_na, ['exclude', 'raise'], 'handle_na')
    left, right = _validate_val_range(val_range)

    values = df[col].astype(float)
    
    # Handle NaN/inf values
    mask_na = values.isna() | values.isin([np.inf, -np.inf])
    if mask_na.any():
        if handle_na == 'raise':
            raise ValueError("NaN or infinite values found in data")
        elif handle_na == 'exclude':
            df = df[~mask_na]

    # Create range mask based on inclusive parameter
    if inclusive == 'both':
        mask = (values >= left) & (values <= right)
    elif inclusive == 'neither':
        mask = (values > left) & (values < right)
    elif inclusive == 'left':
        mask = (values >= left) & (values < right)
    else:  # 'right'
        mask = (values > left) & (values <= right)

    # Apply inside_range logic
    return df[mask if inside_range else ~mask]

# NumPy functionality
## Primary functions
def filter_range_arr(
    arr: np.ndarray,
    col: int,
    val_range: npt.ArrayLike,
    inside_range: bool = True,
    inclusive: str = 'both',
    handle_na: str = 'raise'
    ) -> np.ndarray:
    """
    Select data rows based on value range using binary search.
    Only suitable for SORTED sequences - much faster than filter_range.

    Parameters:
    -----------
    data : np.ndarray
        Input data with sorted values in specified column
    col : int
        Column to select on (must be monotonically increasing/decreasing)
    val_range : tuple or list
        (min_value, max_value) defining the range
    inside_range : bool, default True
        If True, keep values within range
        If False, keep values outside range
    inclusive : str, default 'both'
        Include boundaries: 'both', 'neither', 'left', 'right'
    handle_na : str, default 'raise'
        How to handle NaN/Inf values: 'exclude' or 'raise'.
        If 'exclude', NaN/Inf excluded before range selecting. Returned data will not contain them.
        Pre-filtering is recommended for more complex NA handling strategies

    Returns:
    --------
    numpy.ndarray
        Selected data

    Raises:
    -------
    TypeError
        If inputs have incorrect types
    ValueError
        If the column is not monotonic or if parameters are invalid
        If column is out bounds for array
        
    Notes:
    ------
    Supports open boundaries using NaN values:
    - (NaN, x) means "less than or equal to x"
    - (x, NaN) means "greater than or equal to x"
    - (NaN, NaN) means "select all" if inside_range=True, "select none" if inside_range=False
    """
    # Input validation
    x, col = _validate_array_and_extract_column(arr, col)
    left, right = _validate_val_range(val_range)
    if not isinstance(inside_range, bool):
        raise TypeError("inside_range must be a boolean")
    inclusive = _validate_option(inclusive, ['both', 'neither', 'left', 'right'], 'inclusive')
    handle_na = _validate_option(handle_na, ['exclude', 'raise'], 'handle_na')
    
    mask_na = ~np.isfinite(x)
    if mask_na.any():
        if handle_na == 'raise':
            raise ValueError("NaN or infinite values found in data")
        elif handle_na == 'exclude':
            x = x[~mask_na].copy()
    
    # Create range mask based on inclusive parameter
    mask = _create_range_mask(x, left, right, inclusive)

    # Apply inside_range logic
    mask = mask if inside_range else ~mask
    
    return arr[mask & ~mask_na]

def select_range_arr(
    arr: np.ndarray,
    col: int,
    val_range: npt.ArrayLike,
    inside_range: bool = True,
    inclusive: str = 'both',
    handle_na: str = 'raise'
    ) -> np.ndarray:
    """
    Select data rows based on value range using binary search.
    Only suitable for SORTED sequences - much faster than filter_range.

    Parameters:
    -----------
    data : np.ndarray
        Input data with sorted values in specified column
    col : int
        Column to select on (must be monotonically increasing/decreasing)
    val_range : tuple or list
        (min_value, max_value) defining the range
    inside_range : bool, default True
        If True, keep values within range
        If False, keep values outside range
    inclusive : str, default 'both'
        Include boundaries: 'both', 'neither', 'left', 'right'
    handle_na : str, default 'raise'
        How to handle NaN/Inf values: 'exclude' or 'raise'.
        If 'exclude', NaN/Inf excluded before range selecting. Returned data will not contain them.
        Pre-filtering is recommended for more complex NA handling strategies

    Returns:
    --------
    numpy.ndarray
        Selected data

    Raises:
    -------
    TypeError
        If inputs have incorrect types
    ValueError
        If the column is not monotonic or if parameters are invalid
        If column is out bounds for array
        
    Notes:
    ------
    Supports open boundaries using NaN values:
    - (NaN, x) means "less than or equal to x"
    - (x, NaN) means "greater than or equal to x"
    - (NaN, NaN) means "select all" if inside_range=True, "select none" if inside_range=False
    """
    # Input validation
    x, col = _validate_array_and_extract_column(arr, col)
    left, right = _validate_val_range(val_range)
    if not isinstance(inside_range, bool):
        raise TypeError("inside_range must be a boolean")
    inclusive = _validate_option(inclusive, ['both', 'neither', 'left', 'right'], 'inclusive')
    handle_na = _validate_option(handle_na, ['exclude', 'raise'], 'handle_na')
    
    mask_na = ~np.isfinite(x)
    if mask_na.any():
        if handle_na == 'raise':
            raise ValueError("NaN or infinite values found in data")
        elif handle_na == 'exclude':
            valid_indices = np.where(~mask_na)[0]
            x = x[~mask_na].copy()
    else:
        valid_indices = np.arange(len(arr))
    
    # Check if column is monotonic
    is_monotonic = _check_monotonic_1darray(x) # returns 1, -1, 0 for increasing, decreasing and nonmonotonic
    
    if not is_monotonic:
        raise ValueError(f"Column '{col}' must be monotonically increasing or decreasing")

    is_decreasing = True if is_monotonic == -1 else False
    left_idx, right_idx = _get_range_boundary_indices(x, left, right, inclusive, is_decreasing)

    # Create a mask for the selected range based on original indices
    mask = np.zeros(len(arr), dtype=bool)
    mask[valid_indices[left_idx:right_idx]] = True
    # Apply inside_range logic
    return arr[mask if inside_range else ~mask]

def symmetric_range(lim, dx=1, *, sp=0, exclude_sp=False) -> np.ndarray:
    """Create a symmetric range around a symmetry point with exact steps.

    Parameters
    ----------
    lim : float
        One boundary of the range. The other boundary will be determined
        by symmetry properties.
    dx : float, optional
        Step size (default is 1). Must be positive.
    sp : float, optional
        Symmetry point (default is 0).
    exclude_sp : bool, optional
        If True, exclude the symmetry point from the output (default is False).

    Returns
    -------
    ndarray
        Sorted array of values symmetric around `sp` with step size `dx`.

    Raises
    ------
    ValueError
        If `dx` is not positive.
        If `lim` equals `sp`.

    Examples
    --------
    >>> symmetric_range(2, dx=1)
    array([-2, -1,  0,  1,  2])
    >>> symmetric_range(2, dx=1, sp=1)
    array([0, 1, 2])
    >>> symmetric_range(2, dx=1, exclude_sp=True)
    array([-2, -1,  1,  2])
    """
    if not isinstance(dx, (int, float)) or dx <= 0:
        raise ValueError("Step size 'dx' must be positive")

    if lim == sp:
        raise ValueError("Limit cannot be equal to symmetry point")
    
    lim_sym = -lim + 2 * sp
    lim_pos = lim if lim > lim_sym else lim_sym
    half1 = np.arange(sp, lim_pos + dx/2, dx)
    half1 = half1[half1 <= lim_pos]
    
    if exclude_sp:
        half1 = half1[1:]
        half2 = -half1 + 2 * sp
    else:
        half2 = -half1[1::] + 2 * sp
        
    return np.concatenate((half2[::-1], half1))
    
def symmetrize(seq: npt.ArrayLike) -> npt.ArrayLike: 
    if len(seq) == 0:  # Handle empty sequence
        raise ValueError("Sequence cannot be empty")
    seq = np.asarray_chkfinite(seq)
    return (seq + seq[::-1]) / 2

def antisymmetrize(seq: npt.ArrayLike) -> npt.ArrayLike: 
    if len(seq) == 0:  # Handle empty sequence
        raise ValueError("Sequence cannot be empty")
    seq = np.asarray_chkfinite(seq)
    return (seq - seq[::-1]) / 2

def interpolate(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    x_new: npt.ArrayLike,
    *,
    interp: Callable[[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], npt.ArrayLike] | None = None,
    smooth: Callable[[npt.ArrayLike], npt.ArrayLike] | None = None,
    handle_na: str = 'raise'
    ) -> np.ndarray:
    """
    Interpolate and optionally smooth data.

    Parameters
    ----------
    x : array_like
        Original x coordinates.
    y : array_like
        Original y coordinates.
    x_new : array_like
        Target x coordinates for interpolation.
    interp : callable, optional
        Custom interpolation function with signature f(x, y, x_new) -> y_new.
        If None, uses numpy.interp for linear interpolation.
    smooth : callable, optional
        Smoothing function with signature f(y) -> y_smooth.
        If None, no smoothing is applied.

    Returns
    -------
    array_like
        Interpolated and optionally smoothed y values corresponding to x_new.

    Raises
    ------
    ValueError
        If x and y have different lengths, or if x_new contains values outside
        the range of x, or if provided methods are not callable.

    Examples
    --------
    >>> x = [1, 2, 3]
    >>> y = [1, 4, 9]
    >>> x_new = [1.5, 2.5]
    >>> interpolate(x, y, x_new)
    array([2.5, 6.5])
    """
    handle_na = _validate_option(handle_na, ['exclude', 'raise'], 'handle_na')
    x, y = _validate_xy(x, y, handle_na)
    x_new = np.asarray_chkfinite(x_new)

    if interp is not None:
        if not callable(interp):
            raise ValueError("'interp' must be callable")
    else:
        def interp(x, y, x_new): return np.interp(x_new, x, y)

    if smooth is not None and not callable(smooth):
        raise ValueError("smooth_method must be callable")

    y_new = interp(x, y, x_new)

    if smooth:
        y_new = smooth(y_new)

    return y_new
        
def normalize(y: npt.ArrayLike, by: str | float | int) -> np.ndarray:
    """Normalize array by specified value or method.

    Args:
        y: Input array to normalize
        by: Normalization method ('first', 'mid', 'last') or numeric value

    Returns:
        Normalized numpy array
    """
    y_arr = np.asarray(y)
    if not y_arr.size:
        return y_arr

    if isinstance(by, (int, float)):
        val_norm = by
    elif by == 'first':
        val_norm = y_arr[0]
    elif by == 'mid':
        val_norm = get_mid_elem(y_arr)
    elif by == 'last':
        val_norm = y_arr[-1]
    else:
        raise ValueError(f"Invalid normalization method: {by}")
    
    if val_norm == 0:
        raise ValueError("Cannot normalize by zero")
    return y_arr / val_norm

def quick_fit(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    degrees: int | npt.ArrayLike = 1,
    x_range: npt.ArrayLike | None = None,
    handle_na: str = 'raise'
    ) -> np.ndarray:
    """Quick polynomial/rational fitting using least squares.
    
    Fits y = sum(c[i] * x^degrees[i]) where c[i] are the coefficients to find.
    Supports both standard polynomials and rational functions (negative degrees).
    No normalization is performed by design.
    
    Args:
        x, y: array-like input data
        degrees: int or sequence of numbers, default=1
            If int, fits standard polynomial up to this degree (0 to degrees)
            If sequence, uses these exact degrees (can be negative for rational terms)
        x_range: sequence of [min, max], optional
            Fit using only x values within this range inclusive
        handle_na : str, default 'raise'
            How to handle NaN/inf values: 'exclude' or 'raise'
            
    Returns:
        ndarray: coefficients in ascending degree order if degrees is int,
                or in same order as input degrees if sequence
    
    Examples:
        >>> x = np.linspace(0, 1, 10)
        >>> y = 1 + 2*x  # linear function
        >>> quick_fit(x, y)  # returns approximately [1, 2]
        
        >>> # Rational fit with terms x^-1 and x^1
        >>> quick_fit(x, y, degrees=[-1, 1])
    """
    handle_na = _validate_option(handle_na, ['exclude', 'raise'], 'handle_na')
    MAX_SAFE_DEGREE = 5
    
    # Convert inputs to arrays and validate
    x, y = _validate_xy(x, y, handle_na)
    
    # Validate degrees
    degrees = np.asarray_chkfinite(degrees)
    if not (np.issubdtype(degrees.dtype, np.number) and np.isrealobj(degrees)):
        raise TypeError("'degrees' must be a real number or sequence of numbers")
    if degrees.size == 0:
        raise ValueError("'degrees' cannot be an empty array")
    if degrees.ndim > 1:
        raise ValueError("'degrees' must be 1-dimensional")
    if degrees.size == 1:
        degrees = np.arange(degrees.astype(int) + 1)    
    if len(np.unique(degrees)) != len(degrees):
        raise ValueError("Degrees must be unique")
        
    if np.any(np.abs(degrees) > MAX_SAFE_DEGREE):
        warnings.warn(f"Degrees higher than {MAX_SAFE_DEGREE} may cause numerical instability")
    
    # Validate and apply x_range
    if x_range:
        left, right = _validate_val_range(x_range)
        mask = (x >= left) & (x <= right)
        x, y = x[mask], y[mask]
    
    # Check sufficient points
    n_points = len(x)
    n_coef = len(degrees)
    
    if n_points < n_coef:
        raise ValueError(f"Need at least {n_coef} points for {n_coef} coefficients")
    
    if n_points < 2:
        raise ValueError("Need at least 2 points for fitting")
    
    # Check for zero/near-zero x values with negative degrees
    if any(d < 0 for d in degrees):
        min_abs_x = np.min(np.abs(x))
        if min_abs_x < 1e-10:
            raise ValueError("Near-zero x values detected with negative degrees")
    
    # Create design matrix
    X = np.column_stack([x**d for d in degrees])
    
    # Solve using lstsq
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return coef

def make_curve(
    x: npt.ArrayLike,
    coefs: npt.ArrayLike,
    degrees: npt.ArrayLike | None = None
    ) -> np.ndarray:
    """Create a curve from coefficients and optional degrees.

    Computes y = sum(c[i] * x^degrees[i]) where c[i] are the coefficients.
    If degrees not provided, assumes standard polynomial with ascending degrees.

    Args:
        x: array-like, points to evaluate the curve at
        coefs: array-like, polynomial/rational coefficients
        degrees: sequence of numbers, optional
            Powers for each coefficient. If None, uses [0, 1, ..., len(coef)-1]
            Can include negative values for rational terms

    Returns:
        ndarray: computed y values

    Examples:
        >>> x = np.linspace(0, 1, 10)
        >>> make_curve(x, [1, 2])  # y = 1 + 2x
        >>> make_curve(x, [1, 2], degrees=[-1, 1])  # y = x^-1 + 2x
    """
    try:
        x = np.asarray_chkfinite(x)
        coefs = np.asarray_chkfinite(coefs)
    except ValueError:
        raise ValueError("Inputs contain NaN or inf values")

    if x.size == 0 or coefs.size == 0:
        raise ValueError("Inputs cannot be empty")

    if degrees is None:
        if len(coefs) == 1:  # constant
            return np.full_like(x, coefs[0])
        degrees = range(len(coefs))
    elif len(degrees) != len(coefs):
        raise ValueError("Number of degrees must match number of coefficients")

    # Check for zero/near-zero x with negative degrees
    if any(d < 0 for d in degrees):
        min_abs_x = np.min(np.abs(x))
        if min_abs_x < 1e-10:
            warnings.warn("Near-zero x values detected with negative degrees. "
                          "Results may be numerically unstable.")

    X = np.column_stack([x**d for d in degrees])
    return X @ coefs

def check_monotonic_arr(arr: np.ndarray,
                        col: int | None = None,
                        interrupt: bool = False
                        ) -> int:
    """Check if an array column is monotonic.

    Parameters
    ----------
    arr : np.ndarray
        Input array containing the column to check
    col : int, optional
        Index of the column to check for monotonicity.
        Not required if array is 1D
    interrupt : bool, optional
        If True, raises ValueError for non-monotonic data instead of returning 0,
        by default False

    Returns
    -------
    int
        Monotonicity indicator:
         1 : monotonically increasing
         0 : not monotonic
        -1 : monotonically decreasing

    Raises
    ------
    TypeError
        If inputs have incorrect types
    ValueError
        If column index is outside the array
        If column contains non-numeric data
        If column is empty
        If interrupt=True and the column is not monotonic
    """
    x, col = _validate_array_and_extract_column(arr, col)
    if not np.isfinite(x).all():
        raise ValueError(f"Column '{col}' contains NaN/Inf values")
    is_monotonic = _check_monotonic_1darray(x)
    if interrupt and not is_monotonic:
        raise ValueError(f"Column '{col}' is not monotonic")
    return is_monotonic

def evaluate_std(x: npt.ArrayLike, y: npt.ArrayLike, 
                x_range: npt.ArrayLike | None = None,
                ddof: int = 0, 
                handle_na: str = 'raise') -> float:
    """Evaluate standard deviation of the data approximating it with line.

    Parameters
    ----------
    x : array_like
        Independent variable values.
    y : array_like
        Dependent variable values.
    range_val : array_like, optional
        Range of x values to consider (min, max). If None, uses full range.
    ddof : int, default = 0
        Delta degrees of freedom to use to correct for data preprocessing.
        Std is calculated using np.std(residuals, ddof=2+ddof)

    Returns
    -------
    float
        Standard deviation of residuals after removing linear trend.

    Raises
    ------
    ValueError
        If x and y have different lengths
        If range_val is invalid
        If handle_na is invalid
    TypeError
        If inputs are not numeric arrays.
    """
    handle_na = _validate_option(handle_na, ['exclude', 'raise'], 'handle_na')
    x, y = _validate_xy(x, y, handle_na)
    if x_range:
        left, right = _validate_val_range(x_range)
        mask = (x >= left) & (x <= right)
        x, y = x[mask], y[mask]
    coefs = quick_fit(x, y, degrees=1) # linear fit
    lin_approx = make_curve(x, coefs)
    # Using two degrees of freedom because we evaluated two parameters during linear fit
    return np.std(y - lin_approx, ddof=2 + ddof)
       
## Secondary functions
def get_mid_ind(seq: npt.ArrayLike)-> int:
    if len(seq) == 0:  # Handle empty sequence
        raise ValueError("Sequence cannot be empty")
    return len(seq) // 2  # Use floor division
    
def get_mid_elem(seq: npt.ArrayLike) -> Any:
    if len(seq) == 0:
        raise ValueError("Sequence cannot be empty")
    return seq[len(seq) // 2]

def sym_points(sequence, *, split_point=0, atol=1e-8):
    indices = find_sym_points_indices(sequence, split_point, atol)
    seq = np.array(sequence, dtype=float)
    return seq[indices]

def nonsym_points(sequence, *, split_point=0, atol=1e-8):
    indices = ~find_sym_points_indices(sequence, split_point, atol)
    seq = np.array(sequence, dtype=float)
    return seq[indices]
 
def find_sym_points_indices(sequence, split_point=0, atol=1e-8):
    seq = np.asarray_chkfinite(sequence, dtype=float)
    split_idx = np.searchsorted(seq, split_point)

    left = seq[:split_idx]
    right = seq[split_idx:]

    # Calculate distances once
    left_dist = np.abs(left - split_point)
    right_dist = np.abs(right - split_point)

    # Use broadcasting without creating a new dimension
    # This avoids creating a potentially large intermediate array
    matches = np.abs(left_dist[:, None] - right_dist) <= atol

    # Find first match for each left point using argmax
    # argmax returns first True value, or 0 if none found
    has_match = matches.any(axis=1)
    match_idx = matches[has_match].argmax(axis=1)
    left_idx = np.nonzero(has_match)[0]
    right_idx = np.flip(match_idx + split_idx)
    
    # Construct result array directly
    return np.sort(np.concatenate([left_idx, right_idx]))

# Protected functions
def _validate_val_range(val_range: npt.ArrayLike) -> tuple[float, float]:
    """
    Validate and convert range input to a tuple of (left, right) bounds.
    Supports open boundaries using NaN/inf values:
        - (NaN, x) means "less than or equal to x"
        - (x, NaN) means "greater than or equal to x"
        - (NaN, NaN) means "select all"

    Parameters:
    -----------
    val_range : array_like
        Any sequence of two values defining the range bounds

    Returns:
    --------
    tuple : (left, right)
        Processed boundary values where NaN is converted to -inf/inf
    """
    try:
        # Convert to list to handle various sequence types
        range_list = list(val_range)
        if len(range_list) != 2:
            raise ValueError("Range must contain exactly 2 values")

        left, right = range_list

        # Convert to float/handle numeric types
        left = float(left) if not pd.isna(left) else -np.inf
        right = float(right) if not pd.isna(right) else np.inf
        
        if left > right:
            left, right = right, left
        return (left, right)

    except TypeError:
        raise TypeError("Range values must be numeric or NaN")
    
def _validate_xy(x: npt.ArrayLike, y: npt.ArrayLike, handle_na: str = 'raise') -> tuple[np.ndarray, np.ndarray]:
    """Validate and preprocess x and y input arrays.

    Parameters
    ----------
    x : array_like
        Independent variable values
    y : array_like
        Dependent variable values
    handle_na : str, default 'raise'
        How to handle NaN/inf values: raises error if 'raise`,
        excludes them from arrays if 'exclude'

    Returns
    -------
    x_clean, y_clean : ndarray
        Validated and cleaned input arrays

    Raises
    ------
    ValueError
        If inputs have invalid shapes or contain NaN/inf with handle_na='raise'
    TypeError
        If inputs cannot be converted to numpy arrays
    """
    # Convert inputs to arrays and validate
    x, y = np.asarray(x), np.asarray(y)
    
    mask_na = np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y)
    if mask_na.any():
        n_invalid = np.sum(mask_na)
        if handle_na == 'raise':
            raise ValueError(f"Found {n_invalid} NaN/Inf values in data")
        # handle_na == 'exclude'
        x, y = x[~mask_na], y[~mask_na]
    
    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape")
        
    if x.ndim != 1:
        raise ValueError("x and y must be 1-dimensional")
    return x, y

def _validate_array_and_extract_column(
    arr: np.ndarray,
    col: int | None = None
    ) -> tuple[np.ndarray, int]:
    """
    Validates input array and extracts specified column if 2D array.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array (1D or 2D)
    col : int, optional
        Column index to extract from 2D array. If None and array is 2D,
        defaults to 0. Ignored for 1D arrays.

    Returns
    -------
    x : numpy.ndarray
        1D array extracted from input array
    col : int
        Column index that was used (0 for 1D arrays)

    Raises
    ------
    TypeError
        If arr is not numpy.ndarray or col is not integer/None
    ValueError
        If array dimensions are >2, column index is out of bounds,
        column is empty, or contains non-numeric data

    Examples
    --------
    >>> arr_1d = np.array([1, 2, 3])
    >>> x, col = validate_and_extract_column(arr_1d)
    >>> # Returns original array and col=0

    >>> arr_2d = np.array([[1, 2], [3, 4]])
    >>> x, col = validate_and_extract_column(arr_2d, 1)
    >>> # Returns second column [2, 4] and col=1
    """
    # Type checks
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be an ndarray")
    if col is not None and not isinstance(col, (int, np.integer)):
        raise TypeError("col must be an integer or None")

    # Handle 1D array
    if arr.ndim == 1:
        return arr, 0

    # Handle 2D array
    elif arr.ndim == 2:
        col = 0 if col is None else col
        try:
            x = arr[:, col]
        except IndexError:
            raise ValueError(f"Column index '{col}' is out of bounds for array with shape '{arr.shape}'")

    # Reject higher dimensions
    else:
        raise ValueError(f"Only 1D and 2D arrays are supported; got array with shape '{arr.shape}'")

    # Check for empty column
    if len(x) == 0:
        raise ValueError(f"Column '{col}' is empty")

    # Check for non-numeric data
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError(f"Column '{col}' contains non-numeric data")

    return x, col

def _check_monotonic_1darray(values):
    """Check if sequence is monotonic. Can be not stable due to floating point precision"""
    is_increasing = np.all(np.diff(values) >= 0)
    is_decreasing = np.all(np.diff(values) <= 0)
    if is_increasing:
        return 1
    elif is_decreasing:
        return -1
    return 0

def _get_range_boundary_indices(values, left, right, inclusive, is_decreasing=False):
    # Handle reversed sorting
    if is_decreasing:
        left, right = right, left

    # Adjust search sides based on inclusive parameter
    if inclusive == 'both':
        left_side, right_side = 'left', 'right'
    elif inclusive == 'neither':
        left_side, right_side = 'right', 'left'
    elif inclusive == 'left':
        left_side, right_side = 'left', 'left'
    else:  # 'right'
        left_side, right_side = 'right', 'right'

    # Find boundary indices using binary search
    left_idx = values.searchsorted(left, side=left_side)
    right_idx = values.searchsorted(right, side=right_side)
    return left_idx, right_idx

def _create_range_mask(x: np.ndarray, left: float, right: float, inclusive: str) -> np.ndarray:
    if inclusive == 'both':
        return (x >= left) & (x <= right)
    elif inclusive == 'neither':
        return (x > left) & (x < right)
    elif inclusive == 'left':
        return (x >= left) & (x < right)
    else:  # 'right'
        return (x > left) & (x <= right)
    
def _validate_option(value, allowed_values, param_name):
    """
    Validate if a value is among allowed options.

    Parameters
    ----------
    value : Any
        The value to validate
    allowed_values : set or tuple or list
        Collection of allowed values
    param_name : str
        Name of the parameter being validated, used in error message

    Returns
    -------
    Any
        The validated value

    Raises
    ------
    ValueError
        If value is not in allowed_values

    Examples
    --------
    >>> validate_option('exclude', ['exclude', 'raise'], 'handle_na')
    'exclude'
    >>> validate_option('invalid', ['exclude', 'raise'], 'handle_na')
    ValueError: handle_na must be one of: 'exclude', 'raise'
    """
    if value not in allowed_values:
        options_str = "', '".join(str(v) for v in allowed_values)
        raise ValueError(f"{param_name} must be one of: '{options_str}'")
    return value
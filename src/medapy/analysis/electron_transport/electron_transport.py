from typing import Optional, Union, TextIO

from pathlib import Path
import numpy as np
import numpy.typing as npt
import lmfit

from medapy.utils import misc
from medapy.utils.constants import e


# Public functions
## general
def r2rho(r: npt.ArrayLike, kind: str, t: float, width: float = None, length: float = None) -> np.ndarray:
    """
    Convert resistance to resistivity.

    Parameters
    ----------
    r : array-like
        Resistance values
    kind : str
        Type of measurement ('xx' for longitudinal, 'xy' for Hall)
    t : float
        Sample thickness in meters
    width : float, optional
        Sample width in meters (required for 'xx')
    length : float, optional
        Sample length in meters (required for 'xx')

    Returns
    -------
    np.ndarray
        Resistivity values
    """
    kind = misc._validate_option(kind, ['xx', 'xy'], 'kind')
    r = np.asarray_chkfinite(r)    

    if kind == 'xx':
        if width is None or length is None:
            raise ValueError("width and length are required for longitudinal resistivity")
        return r * (width/length) * t
    else:  # xy
        return r * t

## twoband
def gen_hall2bnd_eq(bands):
    sign1, sign2 = list(map(_band_note_to_sign, bands))
    def xy(H, n1, n2, mu1, mu2):
        numerator = sign1*n1*mu1**2 + sign2*n2*mu2**2 + (sign1*n1 + sign2*n2)*(mu1*mu2*H)**2
        denominator = (n1*mu1 + n2*mu2)**2 + ((sign1*n1 + sign2*n2)*(mu1*mu2*H))**2
        return H/e*np.divide(numerator, denominator)
    return xy

def gen_mr2bnd_eq(bands):
    sign1, sign2 = list(map(_band_note_to_sign, bands))
    def xx(H, n1, n2, mu1, mu2):
        numerator = n1*mu1 + n2*mu2 + (n1*mu2 + n2*mu1)*(mu1*mu2*H**2)
        denominator = (n1*mu1 + n2*mu2)**2 + ((sign1*n1 + sign2*n2)*(mu1*mu2*H))**2
        return np.divide(numerator, denominator)/e
    return xx  

def generate_twoband_eq(kind, bands):
    assert (kind in ('xx', 'xy'))
    assert (bands in ('hh', 'he', 'ee', 'eh'))
    match kind:
        case 'xx':
            return gen_mr2bnd_eq(bands)
        case 'xy':
            return gen_hall2bnd_eq(bands)

def fit_twoband(field: np.ndarray, 
                rho: np.ndarray, 
                p0: tuple[float, float, float, float], 
                *, 
                kind: str, 
                bands: str,
                bounds: Optional[Union[dict, tuple[Optional[npt.ArrayLike], Optional[npt.ArrayLike]]]] = None,
                extension: Optional[Union[tuple, np.ndarray]] = None,
                sigma: Optional[Union[float, npt.ArrayLike]] = None,
                fix_params: Optional[Union[dict, npt.ArrayLike]] = None,
                method: str = 'leastsq',
                fit_kwargs: dict = None,
                report: Union[bool, str, Path, TextIO] = False,
                handle_na: str = 'raise') -> tuple[float, float, float, float]:
    """
    Fit resistivity data with two-band model using lmfit.

    Parameters:
    -----------
    field : array-like
        Magnetic field values in tesla units
    rho : array-like
        Resistivity values in ohm*m units
    p0 : tuple
        Initial guess for (n1, n2, mu1, mu2) in m^-3 and m^2/Vs units, correspondingly
    kind : str
        'xy' for Hall or 'xx' for magnetoresistance
    bands : str
        Band type specification
    bounds : dict or tuple, optional
        Parameter bounds
    extention : tuple or ndarray, optional
        Extension data for fitting
    sigma : float or npt.ArrayLike, optional
        Measurement uncertainties
    fix_params : dict or npt.ArrayLike, optional
        Fixed parameters specification
    method : str
        Fitting method passed to lmfit.minimize
    fit_kwargs : dict
        Additional keywords for fitting
    report : Union[bool, str, Path, TextIO]
        If True, prints fit report to console.
        If string or Path, writes report to specified file in 'a' mode
        If file object, writes report to that file
    handle_na : str, default 'raise'
        How to handle NaN/inf values: 'exclude' or 'raise'
    Returns:
    --------
    tuple[float, float, float, float]
        Fitted parameters (n1, n2, mu1, mu2)
    """
    fit_kwargs = fit_kwargs or {}

    # Input validation
    p0 = np.asarray(p0)
    if len(p0) != 4:
        raise ValueError(f"p0 must be a array-like of 4 initial values; "
                         f"got {len(p0)} values")
    if np.any(p0 <= 0):
        raise ValueError("Initial values must be positive numbers")
    kind = misc._validate_option(kind, ['xx', 'xy'], 'kind')
    bands = _validate_band_notes(bands, 2)
    handle_na = misc._validate_option(handle_na, ['exclude', 'raise'], 'handle_na')
    field, rho = misc._validate_xy(field, rho, handle_na)
    # Validate and prepare extension data
    field_ext, rho_ext = _prepare_twoband_fit_extension(extension)

    # Set up equations based on kind
    if kind == 'xy':
        main_eq = gen_hall2bnd_eq(bands=bands)
        ext_eq = gen_mr2bnd_eq(bands=bands)
    else: # kind == 'xx'
        main_eq = gen_mr2bnd_eq(bands=bands)
        ext_eq = gen_hall2bnd_eq(bands=bands)

    # Create parameters
    params = _create_twoband_fit_parameters(p0, bounds, fix_params)
    
    # Prepare sigma values
    sigmas = _prepare_twoband_fit_sigma(sigma, rho.size, rho_ext.size)
    if sigmas is not None:
        fit_kwargs['kws'] = dict(sigmas=sigmas)

    # Perform the fit
    res = lmfit.minimize(_twoband_fit_objective, params,
                  args=(field, field_ext, rho, rho_ext, main_eq, ext_eq),
                  method=method,
                  **fit_kwargs)

    if report:
        report_text = lmfit.fit_report(res)
        if isinstance(report, bool):
            print(report_text)
        elif isinstance(report, (str, Path)):
            with open(report, 'w') as f:
                print(report_text, file=f)
        else: # TextIO
            print(report_text, file=report)
        
    # Extract and rescale final parameters
    p = res.params
    return _rescale_twoband_params(p['n1'].value, p['n2'].value, 
                                 p['mu1'].value, p['mu2'].value)

# Protected methods
## twoband methods
def _twoband_fit_objective(pars, field, field_ext, rho, rho_ext, eq, eq_ext, sigmas=None):
    """Objective function for fitting."""
    v = pars.valuesdict()
    n1, n2, mu1, mu2 = _rescale_twoband_params(v['n1'], v['n2'], v['mu1'], v['mu2'])

    resid = eq(field, n1, n2, mu1, mu2) - rho
    resid_ext = eq_ext(field_ext, n1, n2, mu1, mu2) - rho_ext
    residuals = np.concatenate((resid, resid_ext))

    if sigmas is not None:
        return residuals/sigmas
    return residuals

def _rescale_twoband_params(n1: float, n2: float, mu1: float, mu2: float) -> tuple[float, float, float, float]:
    """Convert parameters from log scale back to linear scale."""
    return (10**n1, 10**n2, 10**mu1, 10**mu2)

    
def _create_twoband_fit_parameters(p0: npt.ArrayLike, 
                     bounds: Optional[Union[dict, tuple[Optional[npt.ArrayLike], Optional[npt.ArrayLike]]]] = None,
                     fix_params: Optional[Union[dict, npt.ArrayLike]] = None) -> lmfit.Parameters:
    """Create and configure lmfit Parameters object."""
    n1_0, n2_0, mu1_0, mu2_0 = map(np.log10, p0)

    params = lmfit.Parameters()
    params.add('n1', value=n1_0)
    params.add('n2', value=n2_0)
    params.add('mu1', value=mu1_0)
    params.add('mu2', value=mu2_0)
    # Set default bounds if none provided
    if bounds is None:
        default_bounds = {
            'n1': (1e20, 1e28), # m^-3
            'n2': (1e20, 1e28), # m^-3
            'mu1': (0.001, 1),  # m^2/Vs
            'mu2': (0.001, 1)   # m^2/Vs
        }
        bounds = default_bounds

    # Apply bounds
    if isinstance(bounds, dict):
        for param_name, bound in bounds.items():
            if bound is not None:
                lb, ub = bound
                if lb is not None:
                    params[param_name].set(min=np.log10(lb))
                if ub is not None:
                    params[param_name].set(max=np.log10(ub))
    else:
        lb_seq, ub_seq = bounds
        param_names = ['n1', 'n2', 'mu1', 'mu2']

        if lb_seq is not None:
            for param_name, lb in zip(param_names, lb_seq):
                if lb is not None:
                    params[param_name].set(min=np.log10(lb))

        if ub_seq is not None:
            for param_name, ub in zip(param_names, ub_seq):
                if ub is not None:
                    params[param_name].set(max=np.log10(ub))
    # Apply fixed parameters
    if fix_params is not None:
        if isinstance(fix_params, dict):
            for p, val in fix_params.items():
                params[p].set(vary=not val)
        else:
            for p, val in zip(['n1', 'n2', 'mu1', 'mu2'], fix_params):
                params[p].set(vary=not val)
    return params

def _prepare_twoband_fit_extension(extention: Optional[Union[tuple, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Prepare extension data for fitting."""
    if extention is None:
        return np.array([]), np.array([])

    if isinstance(extention, (tuple, list)):
        field_ext, rho_ext = extention
    elif isinstance(extention, np.ndarray):
        field_ext = extention[:, 0]
        rho_ext = extention[:, 1]
    else:
        raise TypeError("Extension must be tuple, list, or numpy array")

    return np.asarray_chkfinite(field_ext), np.asarray_chkfinite(rho_ext)

def _prepare_twoband_fit_sigma(sigma: Optional[Union[float, npt.ArrayLike]], 
                 rho_size: int, 
                 rho_ext_size: int) -> Optional[np.ndarray]:
    """Prepare sigma values for fitting."""
    if sigma is None:
        return None

    try:
        sigma = np.asarray_chkfinite(sigma)
        if sigma.size == rho_size + rho_ext_size:
            return sigma.flatten()
        if sigma.size == 1:
            return np.full(rho_size + rho_ext_size, sigma)
        elif sigma.size == 2:
            sgm, sgm_ext = sigma
            if np.size(sgm) == 1:
                sgm = np.full(rho_size, sgm)
            if np.size(sgm_ext) == 1:
                sgm_ext = np.full(rho_ext_size, sgm_ext)
            return np.concatenate((sgm, sgm_ext))
        elif sigma.size == 0:
            return None
    except Exception as e:
        print(f"Error handling sigma: {e}")
        return None

def _validate_band_notes(bands, number=2):
    """
    Validate band configuration string.

    Parameters
    ----------
    bands : str
        String containing band configuration using 'h' for holes and 'e' for electrons
    number : int, optional
        Expected number of bands, by default 2

    Returns
    -------
    str
        Normalized (lowercase) band configuration string

    Raises
    ------
    TypeError
        If bands is not a string
    ValueError
        If bands length is incorrect or contains invalid characters

    Examples
    --------
    >>> validate_bands("he")
    'he'
    >>> validate_bands("heh", length=3)
    'heh'
    >>> validate_bands("HE")
    'he'
    """
    if not isinstance(bands, str):
        raise TypeError(f"bands must be a string, got {type(bands).__name__}")

    if len(bands) != number:
        raise ValueError(f"bands must be exactly {number} characters long")

    if not all(char in ('h', 'e') for char in bands.lower()):
        raise ValueError("bands can only contain 'h' and 'e' characters (for holes and electrons)")

    return bands.lower()

def _band_note_to_sign(band_note: str) -> int:
    if band_note == 'h':
        return 1
    elif band_note == 'e':
        return -1

def hall_fit_to_str(p, x_unit='T', y_unit='ohm*m', n=None, n_unit='m^-3'):
    str_res = (f'a0 = {p[0]:.2e} {y_unit}\n'
               f'k = {p[1]:.2e} {y_unit}/{x_unit}')
    if n:
        str_res += f'\nn = {n:.2e} {n_unit}'
    return str_res

def twoband_fit_to_str(p, bands):
    b1, b2 = bands
    str_res = (f'n1({b1}) = {p[0]:.2e} m^-3\n'
               f'n2({b2}) = {p[1]:.2e} m^-3\n'
               f'mu1({b1}) = {p[2]:.2e} m^2/V/s\n'
               f'mu2({b2}) = {p[3]:.2e} m^2/V/s')
    return str_res
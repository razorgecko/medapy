from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from medapy import ms_pandas
from medapy.analysis.electron_transport import etr, etr_pandas
from medapy.collection import MeasurementCollection, ContactPair, DefinitionsLoader

ureg = ms_pandas.ureg # to ensure that pint UnitRegistry is the same

# Setup path to folder with data
script_dir = Path(__file__).parent
data_dir = script_dir / 'files'
result_dir = data_dir / 'results'
result_dir.mkdir(exist_ok=True)

# Load default parameter definitions
parameters = DefinitionsLoader().get_all()

# Initialize folder as measurement collection
collection = MeasurementCollection(data_dir, parameters)

# For fitting, we select suitable files by parameters; current magnitude in this example
pair_10mA = ContactPair(1, 5, 'I', 10e-3) # create contact pair I1-5(10mA)

# Filter to select a specific xx and xy files by criteria
files_xx = collection.filter(contacts=[pair_10mA, (20, 21)])
print('Files Rxx', files_xx, sep='\n', end='\n\n')
# Unpack the datafile from collection to work with it direclty
xx_datafile = files_xx.files[0]

files_xy = collection.filter(contacts=[pair_10mA, (20, 40)])
print('Files Rxy', files_xy, sep='\n', end='\n\n')
xy_datafile = files_xy.files[0]

# Alternatively we can set the names manually
# xx_datafile = data_dir / 'a_twoband_test_I1-5(10mA)_V20-21_Rxx.csv'
# xy_datafile = data_dir / 'a_twoband_test_I1-5(10mA)_V20-40_Rxy.csv'

# Read the data with pandas
xx = pd.read_csv(xx_datafile.path, delimiter=',', usecols=[0, 1])
xy = pd.read_csv(xy_datafile.path, delimiter=',')

# Initialize measurement sheets
custom_unit_dict = dict(Ohms='ohm')
xx.ms.init_msheet(translations=custom_unit_dict, patch_rename=True)
xy.ms.init_msheet(translations=custom_unit_dict, patch_rename=True)
xx.ms.rename({'Resistance': 'Resistance_xx'})
xy.ms.rename({'Resistance': 'Resistance_xy'})

# Concatenate different dataframes preserving MS metadata
data = xx.ms.concat(xy)
data.ms.add_labels({'Field': 'H',
                    'Resistance_xx': 'Rxx',
                    'Resistance_xy': 'Rxy'})

# Validate that x axis (Field column) is monotonously increasing
data.etr.ensure_increasing(inplace=True)

# Geometric parameters of the sample
length = 20e-6
width = 40e-6
t = 400e-9

# It's possible to use pint quantities instead
# length = 20 * ureg.micrometer
# width = 40 * ureg('um')
# t = 400e-9 * ureg.meter

# Convert resistance to resistivity
# This will add the resistivity column with name specified in add_col (default - 'rho')
# The name will be modified by attaching '_xx' or '_xy'
# To not add the column pass empty string
# inplace parameter is mimicking pandas and determines whether to modify current dataframe
data.etr.r2rho('xx', col='Rxx', t=t, width=width, length=length,
               new_col='Resistivity_xx', add_label='rho_xx',
               inplace=True)
data.etr.r2rho('xy', col='Rxy', t=t,
               new_col='Resistivity_xy', set_axis='y', add_label='rho_xy',
               inplace=True)

# Make a standard Hall fitting on a range > 11
# If add_col is not empty, fit values will be added to a new column
# add_col is an appendix to the column name used (default - 'linHall')
lin_coefs, _ = data.etr.fit_linhall(x_range=(11, None), set_axis='l', add_label='flin', inplace=True)

# Select a range of data inside or outside a specific x axis range
part_xx = data.ms[['Field', 'rho_xx']]
part_xx = part_xx.etr.select_range((-6, 6), inside_range=False) # keep only data outside the range

# Two-band fitting
bands = 'he' # hole and electron bands
p0 = [1e26, 1e25, 0.015, 0.02] # starting values [n1, n2, mu1, mu2] in SI units
# We can use extension to fit self-consistently two sets of data
# report determines where to print the fitting report
# can be bool, path to file, or opened file. If True, prints to console
# Try to use
# report = result_dir / 'twoband_reports.txt'
p_opt_xy, _ = data.etr.fit_twoband(p0, col='rho_xy', kind='xy', bands=bands,
                                   extension=part_xx, set_axis='f', add_label='f2_xy',
                                   report=True, inplace=True)
p_opt_xx, _ = data.etr.fit_twoband(p0, col='rho_xx', kind='xx', bands=bands,
                                   field_range=(-6, 6), inside_range=False,
                                   extension=(data.ms.x, data.ms.y),
                                   add_label='f2_xx', report=True, inplace=True)

# It would be easier to calculate xx values from p_opt
# but current approach is used to illustrate different set of fit_twoband parameters
# data.etr.calculate_twoband(p_opt_xy, cols='rho_xx', kinds='xx', bands=bands,
#                            add_labels='f2_xx', inplace=True)

# Apply scientific format to specified columns
# the rest will use float_format (default - '%.4f')
cols = ['rho_xx', 'rho_xy', 'flin', 'f2_xy', 'f2_xx']
fmtr = {col: '{:.4E}' for col in cols}
data.ms.save_msheet(result_dir / 'data.csv', formatter=fmtr)

# Plot results
fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(10, 5))
fig.suptitle('Fit results')

ax1.set_title('xx')
ax1.set_xlabel(f'Field ({data.ms.get_unit("H")})')
ax1.set_ylabel(f'rho_xx ({data.ms.get_unit("rho_xx")})')
ax2.set_title('xy')
ax2.set_xlabel(f'Field ({data.ms.get_unit("H")})')
ax2.set_ylabel(f'rho_xy ({data.ms.get_unit("rho_xy")})')

# We can access all the data by axis names assigned during the processing
ax1.plot(data.ms.x, data.ms['rho_xx'], '.', label='data')
ax1.plot(data.ms.x, data.ms['f2_xx'], 'r--', lw=1.2, label='fit')
ax2.plot(data.ms.x, data.ms['rho_xy'], '.', label='data')
ax2.plot(data.ms.x, data.ms['f2_xy'], 'r--', lw=1.2, label='fit')
ax2.plot(data.ms.x, data.ms['flin'], 'k--', lw=1.2, label='fit lin')

ax1.legend()
ax2.legend()
plt.tight_layout()
plt.show()
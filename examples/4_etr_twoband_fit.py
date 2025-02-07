from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from medapy import ureg
from medapy.analysis import electron_transport
from medapy.collection import MeasurementCollection, ContactPair, DefinitionsLoader

# Setup path to folder with data
script_dir = Path(__file__).parent
data_dir = script_dir / 'files'
result_dir = data_dir / 'results'
result_dir.mkdir(exist_ok=True)

# Load default parameter definitions
parameters = DefinitionsLoader().get_all()

# Initialize folder as measurement collection
collection = MeasurementCollection(data_dir, parameters)

# For fittin we select suitable files by current magnitude
pair_10mA = ContactPair(1, 5, 'I', 10e-3) # create contact pair I1-5(10mA)

# Filter to select a specific xx and xy files by criteria
files_xx = collection.filter(contacts=[pair_10mA, (20, 21)])
print('Files Rxx', files_xx, sep='\n', end='\n\n')
# Unpack the datafile from collection to work with it direclty
xx_datafile = files_xx.files[0]

files_xy = collection.filter(contacts=[pair_10mA, (20, 40)])
print('Files Rxy', files_xy, sep='\n', end='\n\n')
xy_datafile = files_xy.files[0]

# Alternatively we could have manually set the names
# xx_datafile = data_dir / 'a_twoband_test_I1-5(10mA)_V20-21_Rxx.csv'
# xy_datafile = data_dir / 'a_twoband_test_I1-5(10mA)_V20-40_Rxy.csv'

# Read the data with pandas
xx = pd.read_csv(xx_datafile.path, delimiter=',', usecols=[0, 1])
xy = pd.read_csv(xy_datafile.path, delimiter=',')

# Initialize measurement sheets
custom_unit_dict = dict(Ohms='ohm')
xx.ms.init_msheet(translations=custom_unit_dict, patch_rename=True)
xy.ms.init_msheet(translations=custom_unit_dict, patch_rename=True)


# Validate that x axis (Field column) is monotonously increasing
xx.etr.ensure_increasing(inplace=True)
xy.etr.ensure_increasing(inplace=True)

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
xx.etr.r2rho('xx', t, width=width, length=length, inplace=True, set_axis='y', add_label='rho')
xy.etr.r2rho('xy', t, add_col='custom_rho', inplace=True, set_axis='y', add_label='rho')

# Make a standard Hall fitting on a range > 11
# Currently the range is not an optional parameter, to use whole range pass (None, None)
# This adds the fit result to a new column
# add_col is an appendix to the column name of y axis (default - 'linHall')
lin_coefs = xy.etr.fit_linhall((11, None), set_axis='l', add_label='flin')

# Select a range of data inside or outside a specific x axis range
part_xx = xx.etr.select_range((-6, 6), inside_range=False)

# Two-band fitting
bands = 'he' # hole and electron bands
p0 = [1e26, 1e25, 0.015, 0.02] # starting values [n1, n2, mu1, mu2] in SI units
# We can use extension to fit self-consistently two sets of data
# report determines where to print the fitting report
# can be bool, path to file, or opened file. If True, prints to console
# Try to use
# report = result_dir / 'twoband_reports.txt'
p_opt_xy = xy.etr.fit_twoband(p0, kind='xy', bands=bands, extension=part_xx, report=True,
                              set_axis='f', add_label='f2bnd')
p_opt_xx = xx.etr.fit_twoband(p0, kind='xx', bands=bands, field_range=(-6, 6), inside_range=False,
                              extension=xy, report=True, set_axis='f', add_label='f2bnd')


# Apply scientific format to specified columns
# the rest will use float_format (default - '%.4f')
cols = ['rho', 'f2bnd', 'flin']
fmtr_xx = {col: '{:.4E}' for col in cols[:-1]}
fmtr_xy = {col: '{:.4E}' for col in cols}
xx.ms.save_msheet(result_dir / 'xx.csv', formatter=fmtr_xx)
xy.ms.save_msheet(result_dir / 'xy.csv', formatter=fmtr_xy, float_format=None)

# Plot results
fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(10, 5))
fig.suptitle('Fit results')

ax1.set_title('xx')
ax1.set_xlabel('Field (T)')
ax1.set_ylabel('rho_xx (ohm*m)')
ax2.set_title('xy')
ax2.set_xlabel('Field (T)')
ax2.set_ylabel('rho_xy (ohm*m)')

# We can access all the data by axis names assigned during the processing
ax1.plot(part_xx.ms.x, part_xx.ms.y, '.', label='data')
ax2.plot(xy.ms.x, xy.ms.y, '.', label='data')
ax1.plot(xx.ms.x, xx.ms.f, 'r--', lw=1.2, label='fit')
ax2.plot(xy.ms.x, xy.ms.f, 'r--', lw=1.2, label='fit')
ax2.plot(xy.ms.x, xy.ms.l, 'k--', lw=1.2, label='fit lin')

ax1.legend()
ax2.legend()
plt.tight_layout()
plt.show()
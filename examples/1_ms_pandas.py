import pandas as pd
import pint_pandas
from medapy import ms_pandas


# Create sample DataFrame
df = pd.DataFrame({'Field (Oe)': [1, 2, 3],
                    'Current (uA)': [10, 10, 10],
                    'Voltage (mV)': [0.05, 0.1, 0.15],
                    'Resistance (Ohm)': [5, 10, 15],
                    'Resistivity (uohm cm)': [20, 40, 60],
                    'Magnetoresistance': [1.01, 1.03, 1.07] # dimensionless
                    })
custom_unit_dict = dict(Ohm='ohm') # to map units written not as in pint
df.ms.init_msheet(translations=custom_unit_dict, patch_rename=True) # initialize measurement sheet
print('\nInitialized MSheet:')
print(df.ms) # print msheet to see units, axes, and labels


# Add some labels
# allowed to have several labels for the same column
df.ms.add_labels({'Field': ['H', 'B'],
                  'Resistance': 'R',
                  'Resistivity': 'rho',
                  'Voltage': 'V',
                  'Current': 'I'
                  })
df.ms.add_label('Field', 'Champ')


# Axis assignment
# By default, x, y, z axes are assigned to the first three columns
# {'x': 'Field', 'y': 'Current', 'z': 'Voltage'}
df.ms.set_as_axis('R', 'u') # add new axis
df.ms.set_as_axis('rho', 'y') # reassign y axis to rho
df.ms.set_as_axis('Voltage', 'x', swap=True) # assign axis and swap if both exist
print('\nMSheet with labels and new axes:')
print(df.ms)


# Various methods to access data
fld = df['Field'] # standard df access
fld = df.ms['Field'] # MS access by column
fld = df.ms["H"] # MS access by label
rho = df.ms.y # MS access by axis
fld_r = df.ms[["H", "Resistance"]] # get MS with only specified columns
print('\nSlice of MSheet:')
print(fld_r.ms)


# Unit conversion
# Option 1: manually calculate values and change unit
df.ms['V'] = df.ms['V'] * 1000
df.ms.set_unit('V', 'uV')
# Option 2: use dedicated function
df.ms.convert_unit('H', 'T', contexts='Gaussian')
# contexts is an optional parameter for pint package
# here it is required to be able to convert Oe to T
# conversion rate will be 1 T = 9999.9999972579 Oe
print('\nMSheet after unit conversions:')
print(df.ms)


# Calculations preserving units (pint_pandas is required)
# It is possible to get a column as an array with units
# and use it for computations of additional quantities
crnt = df.ms.wu('I') # 'wu' means 'with units'
volt = df.ms.wu('V')
df.ms['Conductance'] = crnt / volt
print('\nMSheet with conductance column:')
print(df.ms)
df.ms.convert_unit('Conductance', 'siemens') # to force conversion from uA/uV to siemens

# # units, axes, and labels can be accesed as attributes
# units = df.ms.units # map column name -> unit
# axes = df.ms.axes # map axis -> column name
# labels = df.ms.labels # map label -> column name

# # get unit, axis, and labels for a specific column
# unit = df.ms.get_unit('H')
# axis = df.ms.is_axis('H')
# labels = df.ms.get_labels('Field') # mandatory use of column name

# # Check that pandas operations preserve MSheet settings
# df2 = df.rename(columns={'Field': 'FIELD'})
# print('\nNew MSheet got by renaming:')
# print(df2.ms)
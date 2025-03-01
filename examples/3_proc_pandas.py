import pandas as pd
import numpy as np
from medapy import ms_pandas
from medapy.analysis import proc_pandas


# Create sample DataFrame
df = pd.DataFrame({'Field (Oe)': [5, 3, 1, 0, -1, -3, -5],
                    'Current (uA)': [10, 10, 10, 10, 10, 10, 10],
                    'Voltage (mV)': [103, 49, 22, 10, 18, 51, 97],
                    'Resistance (Ohm)': [10300,  4900,  2200, 1000,  1800,  5100,  9700],
                    'Resistivity (Ohm*cm)': [0.2575, 0.1225, 0.055, 0.025, 0.045, 0.1275, 0.2425],
                    })

custom_unit_dict = dict(Ohm='ohm') 
df.ms.init_msheet(translations=custom_unit_dict, patch_rename=True)

# Add labels
df.ms.add_labels({'Field': 'H', 'Resistance': 'R',
                  'Voltage': 'V', 'Current': 'I',
                  'Resistivity': 'rho'})

# Set y axis
df.ms.set_as_y('R')

# Display current DataFrame
print('\nOriginal MSheet:')
print(df.ms)


# Functionality
# is_monotonic = df.etr.check_monotonic() # check if monotonic manually
df.proc.ensure_increasing(inplace=True)
print('\nMSheet forced to monotonic increasing:')
print(df.ms)

# Select range
df1 = df.proc.select_range((-1, 3), inplace=False)
print('\nMSheet in range:')
print(df1.ms)

# Normalizing
# calling functions with cols=None defaults to y-axis column
df.proc.normalize(by='mid', append='norm', inplace=True)
print('\nMSheet after normalizing:')
print(df.ms)

# Symmetrizing
df.proc.symmetrize(cols=['R', 'rho', 'V'], inplace=True)
print('\nMSheet after symmetrization:')
print(df.ms)

# Interpolation
fld = np.linspace(0, 4, 6)
df2 = df.proc.interpolate(fld, cols=['R', 'rho', 'V'], inplace=False)
print('\nMSheet with interpolated values:')
print(df2.ms)
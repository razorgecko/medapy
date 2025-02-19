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
# print(df.ms) # check initialization

# Add some labels
df.ms.add_label('Field', 'H')
# allowed to have several labels for the same column
df.ms.add_label('Field', 'B') 
df.ms.add_label('Resistance', 'R')
df.ms.add_label('Resistivity', 'rho')
df.ms.add_label('Voltage', 'V')
df.ms.add_label('Current', 'I')

# Check various methods to access data
print(f'Standard df access:\n{df["Field"]}\n')
print(f'MS access by column:\n{df.ms["Field"]}\n')
print(f'MS access by label:\n{df.ms["H"]}\n')
print(f'MS access several columns:\n{df.ms[["H", "R"]]}\n')
print(f'MS access by axis:\n{df.ms.y}\n')

# Check axis reassignment
# By default, x, y, z axes are assigned to the first three columns
# {'x': 'Field', 'y': 'Current', 'z': 'Voltage'}
print(f'Default axes: {df.ms.axes}')
df.ms.set_as_axis('u', 'R') # add new axis
df.ms.set_as_axis('y', 'rho') # reassign y axis to rho
df.ms.set_as_axis('x', 'Voltage', swap=True) # assign axis and swap if both exist
print('=========Original df:')
print(df.ms)
print()

# Convert columns to units
# Option 1: manually calculate values and change units
df.ms['V'] = df.ms['V'] * 1000
df.ms.set_unit('V', 'uV')
# Option 2: use dedicated function
df.ms.convert_unit('H', 'T', contexts='Gaussian')
# contexts is an optional parameter for pint package
# here it is required to be able to convert Oe to T
# conversion rate will be 1 T = 9999.9999972579 Oe
print(df)
print(df.ms)

# It is possible to get a column as an array with units
# and make some operations with them
# method wu (with units) is used
crnt = df.ms.wu('I')
volt = df.ms.wu('V')
df.ms['Conductivity'] = crnt / volt
print(df.ms)


# # Properties are copied correctly on pandas operations
# df2 = df.rename(columns={'Field': 'NewField'})
# print('=========New df after rename preserves data:')
# print(df2.ms)
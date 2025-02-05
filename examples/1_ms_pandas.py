import pandas as pd
import medapy


# Create sample DataFrame
df = pd.DataFrame({'Field (Oe)': [1, 2, 3],
                    'Current (uA)': [10, 10, 10],
                    'Voltage (mV)': [0.05, 0.1, 0.15],
                    'Resistance (Ohm)': [5, 10, 15],
                    'Resistivity (uohm cm)': [20, 40, 60]
                    })
custom_unit_dict = dict(Ohm='ohm') # to map units written not as in pint
df.ms.init_msheet(translations=custom_unit_dict, patch_rename=True) # initialize measurement sheet
print(df.ms) # check initialization

# Add some labels
df.ms.add_label('H', 'Field')
df.ms.add_label('B', 'Field')
df.ms.add_label('R', 'Resistance')
df.ms.add_label('rho', 'Resistivity')

# Check various methods to access data
# Accessing by label returns pd.Series with pint units
# Accessing by axis return pd.Series cleared of units
# For processing, it's preferred to pass data as axes
print(f'Standard df access:\n{df["Field"]}\n')
print(f'MS access by column:\n{df.ms["Field"]}\n')
print(f'MS access by label:\n{df.ms["H"]}\n')
print(f'MS access by axis:\n{df.ms.y}\n')
print(f'MS access with get() by column:\n{df.ms.get("Field")}\n')
print(f'MS access with get() by label:\n{df.ms.get("H")}\n')

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

# Properties are copied correctly on pandas operations
df2 = df.rename(columns={'Field': 'NewField'})
print('=========New df after rename preserves data:')
print(df2.ms)
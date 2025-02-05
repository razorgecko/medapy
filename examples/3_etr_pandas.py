import pandas as pd
import numpy as np
import medapy
from medapy.analysis import electron_transport

# ureg = ms_pandas.ureg
# Create sample DataFrame
df = pd.DataFrame({'Field (Oe)': [3, 2, 1, 0, -1],
                    'Current (uA)': [10, 10, 10, 10, 10],
                    'Voltage (mV)': [0.05, 0.1, 0.15, 0.2, 0.25],
                    'Resistance (Ohm)': [5, 10, 15, 20, 25],
                    })
custom_unit_dict = dict(Ohm='ohm') 
df.ms.init_msheet(translations=custom_unit_dict, patch_rename=True)

# Add labels
df.ms.add_label('H', 'Field')
df.ms.add_label('R', 'Resistance')

# Set y axis
df.ms.set_as_y('R')

# Display current DataFrame
print(20*'=', ' Original DataFrame')
print(df)
print(df.ms)
print()

# Functionality tests
print(20*'=', ' Force DataFrame to monotonic increase')
is_monotonic = df.etr.check_monotonic()
df.etr.ensure_increasing(inplace=True)
print(is_monotonic)
print(df)
print(df.ms)
print()

print(20*'=', ' Normalize DataFrame and select range')
df.etr.normalize(by='last', inplace=True)
df.etr.select_range((0,3), inplace=True)
print(df)
print(df.ms)
print()

print(20*'=', ' New DataFrame with interpolated values')
fld = np.linspace(0, 3, 5)
df1 = df.etr.interpolate(fld)
df1.ms.add_label('H', 'Field')
df1.ms.add_label('R', 'Resistance')
print(df1)
print(df1.ms)
print()

print(20*'=', ' Symmetrize column')
df1.etr.symmetrize(inplace=True)
print(df1)
print(df1.ms)
print()

print(20*'=', ' Convert R to rho')
t = 100# * ureg('nm')
w = 2# * ureg('cm')
l = 2# * ureg('cm')
df1.etr.r2rho('xx', t, w, l, inplace=True)
print(df1)
print(df1.ms)
print()


print(20*'=', ' Copy DataFrame')
df2 = df1.copy()
print(df2)
print(df2.ms)
print()

print(20*'=', ' Dequantify DataFrame')
df2 = df2.pint.dequantify()
print(df2)
print(df2.ms)
print(df2.columns)
print(df2.dtypes)
print(df2['Field'])

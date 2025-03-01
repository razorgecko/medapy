import pandas as pd
from medapy import ms_pandas
from medapy.analysis import electron_transport
import matplotlib.pyplot as plt


ureg = ms_pandas.ureg # to ensure that pint UnitRegistry is the same

# Create sample DataFrame
df = pd.DataFrame({'Field (Oe)': [5, 3, 1, 0, -1, -3, -5],
                    'Current (uA)': [10, 10, 10, 10, 10, 10, 10],
                    'Voltage (mV)': [103, 49, 22, 10, 18, 51, 97],
                    'Resistance (Ohm)': [10300,  4900,  2200, 1000,  1800,  5100,  9700],
                    })

custom_unit_dict = dict(Ohm='ohm') 
df.ms.init_msheet(translations=custom_unit_dict, patch_rename=True)

# Add labels
# Add labels
df.ms.add_labels({'Field': 'H', 'Resistance': 'R',
                  'Voltage': 'V', 'Current': 'I'})

# Set y axis
df.ms.set_as_y('R')

# Display current DataFrame
print('\nOriginal MSheet:')
print(df.ms)


# Functionality
# etr_pandas is a subclass of proc_pandas
# that means all proc methods can be called from etr too
df.etr.ensure_increasing(inplace=True)
print('\nMSheet forced to monotonic increasing:')
print(df.ms)

# In addition, etr allows:
# Convert resistance to resistivity 
t = 50 * ureg('nm') # set geometric parameters
w = 10 * ureg('cm')
l = 2 * ureg('cm')

df.etr.r2rho('xx', col='R', t=t, width=w, length=l, new_col='Resistivity', add_label='rho', inplace=True)
print('\nMSheet with calculated resistivity:')
print(df.ms)


df.ms.convert_unit('rho', 'ohm*cm') # vonvert from m to cm for more convenient display
df.ms.set_as_y('rho')

# Make a linear Hall fit
# fit_linhall returns coefficients and resulting DataFrame or None if inplace=True
lin_coefs, _ = df.etr.fit_linhall(col='rho', x_range=(1, None),
                                  set_axis='l', add_label='flin',
                                  inplace=True)
print(f'\nMSheet with linear Hall fit (coefs={lin_coefs}):')
print(df.ms)

# Make a two-band Hall fit
bands = 'he' # hole and electron bands
p0 = [1e26, 1e25, 0.015, 0.02] # starting values [n1, n2, mu1, mu2] in SI units
p_opt, _ = df.etr.fit_twoband(p0, kind='xx', bands=bands, report=True,
                              set_axis='f', add_label='f2bnd', inplace=True)

print(f'\nMSheet with two-band Hall fit (coefs={[f"{x:.2E}" for x in p_opt]}):')
print(df.ms)

fig, ax = plt.subplots(layout='constrained')
ax.set_title('ETR accessor example')
ax.set_xlabel(f'Field ({df.ms.get_unit("H")})')
ax.set_ylabel(f'Resistivity ({df.ms.get_unit("rho")})')
ax.plot(df.ms.x, df.ms['rho'], 'o', label='data')
ax.plot(df.ms.x, df.ms['flin'], label='linear fit')
ax.plot(df.ms.x, df.ms['f2bnd'], label='two-band fit')
ax.legend()
plt.show()
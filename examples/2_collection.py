from pathlib import Path

from medapy.collection import (MeasurementCollection, ParameterDefinition,
                               MeasurementFile, ContactPair, DefinitionsLoader)


# helper function to print files from collection
def print_files(fs, header=None, end=None):
    if header:
        print(header)
        print('-'*len(header))
    for (i, f) in enumerate(fs):
        print(f'{i:2}: {f.path.name}')
    if end:
        print('-'*len(end))
        print(end)
    print()
    
# Create parameter definitions inside code
# example:
#   long_names: [name1, name2]
#   short_names: [n1, n2]
#   units: [unit1, unit2]
#   special_values:
#     value_name1: value1
#     value_name2: value
#   patterns:
#     fixed: "{SNAME}--{VALUE}{UNIT}" 
#     sweep: "{LNAME}sweep|sweep{LNAME}"
#     range: "{NAME}{VALUE}to{VALUE}{UNIT}"

# valid replacers for pattern:
#   {NAME} - union of long and short names
#   {SNAME} - short names
#   {LNAME} - long names
#   {VALUE} - number (integer, with decimal part, or in scientific notation)
#   {UNIT} - units


field_param = ParameterDefinition(name_id = 'field',
    long_names=['field', 'champ', 'Field', 'Champ'],
    short_names=['B', 'H'],
    units=['T', 'Oe', 'G', 'mT']
)
temp_param = ParameterDefinition(
    name_id = 'temperature',
    long_names=['temperature', 'Temperature'],
    short_names=['T'],
    units=['K', 'mK']
)

# Create test measurement file representation
testname = "sample_V1-5(1e-3A)_V3-7_V2-8_V4_V6_I11_sweepField_B-14to14T_T=3.0K_date.csv"
testfile = MeasurementFile(testname, parameters=[field_param, temp_param])

# Print states of parameters parsed from testname 
print(list(map(str, testfile.parameters.values())), end='\n\n')

# Load default parameter definitions
parameters = DefinitionsLoader().get_all()
print(parameters, end='\n\n') # to check loaded parameter definitions

# Set path to folder with files
path = Path(__file__).parent.absolute() / 'files'

# Initialize folder as measurement collection
collection = MeasurementCollection(path, parameters)
print_files(collection, header='Full collection')

# Filter sweep direction
files_down = collection.filter(sweep_direction='down')
print_files(files_down, header='Sweep -- down')

# Filter temperature range from 2 to 7K
# To avoid storing the whole list of files we can use generator instead
files_T2_7 = collection.filter_generator(temperature=(2, 7))
print_files(files_T2_7, header='Temperature 2-7K')

# Filter contact pair
files_V2_6 = collection.filter_generator(contacts=(2, 6))
print_files(files_V2_6, header='Contact pair 2-6')

# Filter contact pair with particular polarization
pair = ContactPair(1, 5, 'I', 1e-6) # create contact pair I1-5(1uA)
# Alternatively it can be created by parsing a string
# pair = ContactPair()
# pair.parse_contacts('I1-5(1uA)')
files_1uA = collection.filter_generator(contacts=pair)
print_files(files_1uA, header='Contact pair 1-5, I=1uA')

# Combine several filters
files_specific = collection.filter_generator(
    contacts=[pair, (3, 7)],
    polarization='I',  # current mode
    temperature=(2, 10),
    position=[45, 'IP']
)
print_files(files_specific, header='Files filtered in a specific way')


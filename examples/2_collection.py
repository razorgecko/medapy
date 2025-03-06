from pathlib import Path

from medapy.collection import (MeasurementCollection, DefinitionsLoader,
                               ParameterDefinition, MeasurementFile, ContactPair)


# helper function to print files from collection
def print_files(fs, n, header=None, end=None):
    if header:
        print(header)
        print('-'*len(header))
    
    if isinstance(fs, MeasurementCollection):
        fs_len = len(fs)
        fs.head(n)
    else:
        for (i, f) in enumerate(fs):
            if i < n:
                print(f'{i:2}: {f.path.name}')
            
        fs_len = i + 1
    print(f'Total number of elements: {fs_len}')
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
testname = "sample_V1-5(1e-3V)_V3-7_V2-8_V4_V6_I11_sweepField_B-14to14T_T=3.0K_date.csv"
testfile = MeasurementFile(testname, parameters=[field_param, temp_param])

testname2 = "sample_I1-5(1e-6A)_V3-7_V2-7_V4_V7_I12_sweepField_B-14to14T_T=3.0K_date.csv"
testfile2 = MeasurementFile(testname2, parameters=[field_param, temp_param])

# Merge two files
# If strict mode, checks that parameters of files are equal 
testfile3 = testfile.merge(testfile2, strict_mode=True)

# Generate filename based on contacts configuration and parameters
# To change separator between name parts, use 'sep' parameter (default - '_')
f = testfile3.generate_filename(prefix='sample', postfix='merged', ext='csv')
print(f)

# Print states of parameters parsed from testname 
print(*list(map(repr, testfile.parameters.values())), sep='\n', end='\n\n')

# Load default parameter definitions
parameters = DefinitionsLoader().get_all()
print(*parameters, sep='\n', end='\n\n') # to check loaded parameter definitions

# Set path to folder with files
path = Path(__file__).parent.absolute() / 'files'

# Initialize folder as measurement collection
collection = MeasurementCollection(path, parameters)
# To see the content of collection we can use print
# print(collection) # uncomment this line

# Similarly to pandas we can also use 'head' and 'tail' methods
collection.head(6) # default 5
collection.tail()
print()


# Collections in many ways behave like lists
# we can add them together, append files, extend from iterables
print(f"Collection length initially: {len(collection)}")

# Get the file from the collection at the given position
meas_f = collection[-1]
# Append file to the collection
collection.append(meas_f)
print(f"Collection length after 'append': {len(collection)}")

# Remove the item at the given position in the collection
# the function returns removed element
meas_f2 = collection.pop() # meas_f == meas_f2
print(f"Collection length after 'pop': {len(collection)}")

# To copy collection call 'copy' method
collection2 = collection.copy()
print(f"Copied collection equals collection: {collection == collection2}")

# Collection can be extended from other collection or iterable
collection2.extend(collection2.copy())
print(f"Copied collection length after 'extend': {len(collection2)}")

# Add two collections
# unlike extend this also adds parameter definitions
collection2 = collection[:5] + collection[-5:]
print(f"New collection length: {len(collection2)}")
print()

# To access results of parameters parsing use 'state_of' method
# Call the method with 'name_id' of required parameter
state = meas_f.state_of('magnetic_field')

# Returned values is the 'ParameterState' of the parameter
print(state)

# The values can be accessed via attributes
print(f"Parameter is swept: {state.is_swept}") # value, min, max, is_swept, sweep_direction

# sweep and range are additional attributes
# they return tuples of related values or None if not available
print(f"Sweep tuple: {state.sweep}") # (min, max, sweep_direction)
print(f"Range tuple: {state.range}") # (min, max)
print()


# To filter data in collection use filter
# Filter sweep direction
# Increasing sweep options: 'inc', 'up', 'increasing'
# Decreasing sweep options: 'dec', 'down', 'decreasing'
files_down = collection.filter(sweep_direction='dec')
print_files(files_down, n=5, header='Sweep -- down')

# Filter temperature range from 2 to 7K
# To avoid storing the whole list of files we can use generator instead
files_T2_7 = collection.filter_generator(temperature=(2, 7))
print_files(files_T2_7, n=5, header='Temperature from 2-7K')

# Filter contact pair
files_V2_6 = collection.filter_generator(contacts=(2, 6))
print_files(files_V2_6, n=5, header='Contact pair 2-6')

# Filter contact pair with particular polarization
pair = ContactPair(1, 5, 'I', 1e-6) # create contact pair I1-5(1uA)
# Alternatively it can be created by parsing a string
# pair = ContactPair()
# pair.parse_contacts('I1-5(1uA)')
files_1uA = collection.filter_generator(contacts=pair)
print_files(files_1uA, n=5, header='Contact pair 1-5, I=1uA')

# Combine several filters
files_specific = collection.filter_generator(
    contacts=[pair, (3, 7)],
    polarization='current',
    sweep_direction='inc', 
    temperature=(2, 10),
    position=[45, 'IP'],
)
print_files(files_specific, n=5, header='Files filtered in a specific way')
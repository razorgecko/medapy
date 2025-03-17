# TODO
## General
- [ ] Write tests
- [ ] Consistent explanatory docstrings (NumPy style)
- [ ] Isolate validations, errors and warnings in separate files
- [ ] Consistent input checks across files
- [ ] Usage examples
- [ ] Optimize imports
- [ ] Consistent typing (pipe style)
- [x] Package installation

## ms_pandas
- [x] Return a proper ms df when slicing existing one
- [ ] Optionality to write metadata (stored in _obj.attrs['meta']) in a file before the data
- [ ] Methods 'to_preferred_units' and 'set_preferred_units'

## collection
- [x] Methods 'append' and 'extend' for MeasurementCollection mimicking list
- [ ] Sort files in collection
    - [x] by parameter value
    - [ ] by sweep direction
- [x] Concatenate mfiles (check equal parameters, append contacts)
    - [ ] sort contact pairs before merging (polarization first, then by contacts values; `__gt__`, `__lt__`)
    - [ ] allow to drop polarization magnitude
- [x] Generate filename based on the state (accept sample name, prefix, postfix, extension)
- [ ] Dump/load collection to/from db
- [ ] Move/copy files from collection to
- [ ] Link dataframes to file objects in collection (?)
- [ ] SI prefix recognition and conversion when parsing units in Parameter
- [ ] Reading file metadata to fill parameters as alternative to parsing filename
- [ ] Datetime stamps for MeasurementFile (from name or system metadata)
- [ ] Consider storing parameter definitions in yaml instead of json (PyYAML package is already installed with some of the dependencies)
- [ ] 'group_by' method to divide collection to subcollections

## accessors
- [x] Split 'etr' to 'ppc' (post-process?) and 'etr' (?)
- [x] Accept columns to interpolate in etr.interpolate 
- [x] Preserve axes and labels in etr.interpolate result (via proper slicing from 'ms_pandas'?)
- [x] Parameter 'set_y' to set newly added column as y (?)
- [ ] Add moving average
- [ ] Units check in etr.fit_twoband
- [ ] Save fit results to dataframe metadata (_obj.attrs) in etr

## misc
- [x] Filter_range_arr
- [ ] Moving average
# TODO
## General
- [ ] Write tests
- [ ] Consistent explanatory docstrings (NumPy style)
- [ ] Isolate validations, errors and warnings in separate files
- [ ] Consistent input checks across files
- [ ] Usage examples
- [ ] Optimize imports
- [ ] Consistent typing (pipe style)
- [ ] Package installation

## ms_pandas
- [ ] Return a proper ms df when slicing existing one
- [ ] Optionality to write metadata (stored in _obj.attrs['meta']) in a file before the data
- [ ] Methods 'to_preferred_units' and 'set_preferred_units'

## collection
- [ ] Methods 'append' and 'extend' for MeasurementCollection mimicking list
- [ ] SI prefix recognition and conversion when parsing units in Parameter
- [ ] Reading file metadata to fill parameters
- [ ] Datetime stamps for MeasurementFile (from name or system metadata)
- [ ] Consider storing parameter definitions in yaml instead of json (PyYAML package is already installed with some of the dependencies)

## accessors
- [ ] Make range for etr.fit_linhall optional
- [ ] Split 'etr' to 'ppc' (post-process?) and 'etr' (?)
- [ ] Accept columns to interpolate in etr.interpolate 
- [ ] Preserve axes and labels in etr.interpolate result (via proper slicing from 'ms_pandas'?)
- [ ] Parameter 'set_y' to set newly added column as y (?)
- [ ] Units check in etr.fit_twoband
- [ ] Save fit results to dataframe metadata (_obj.attrs) in etr

## misc
- [ ] Filter_range_arr
# Magrathea

*Python* codes for constructing planetary models for use with the *NormalModes* library.

## Dependencies

```
pip install --upgrade gmsh
pip install pygmsh
```

## Usage

```
python3 build_spherical.py /path/to/input_file.txt
```

where `input_file.txt` has the following format:

```
input_dir /path/to/input_dir/
output_dir /path/to/output_dir/
tet_max_vol 1.0E7
pOrder 1
is_ellipsoidal 1
use_gravity 0
```

The first column gives the names of the variables to help the user remember them, and this column is ignored by the code. The values in the second column correspond to the following variable names:

* `tet_max_vol` is the maximum allowed size of the tetrahedral mesh element, in km<sup>3</sup> 
* `pOrder` is the finite-element order of the elements (1 or 2, i.e. linear or quadratic)
* `is_ellipsoidal` is 0 if the model is spherical, 1 if it is ellipsoidal.
* `use_gravity` is 1 if gravity should be calculated, 0 otherwise.

The sequence of the lines in the input file should not be changed.

## Storage

Note `-l` flag for symlinks

```
rsync -rvhl tg857131@stampede2.tacc.utexas.edu:/scratch/06414/tg857131/Magrathea/ /Volumes/stoneley5TB/all/Magrathea/v4
```

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
/path/to/input_dir/
/path/to/output_dir/
tet_max_vol
order
is_ellipsoidal
```

where

* `tet_max_vol` is the maximum allowed size of the tetrahedral mesh element, in km<sup>3</sup> 
* `order` is the finite-element order of the elements (1 or 2, i.e. linear or quadratic)
* `is_ellipsoidal` is 0 if the model is spherical, 1 if it is ellipsoidal.

## Storage

Note `-l` flag for symlinks

```
rsync -rvhl tg857131@stampede2.tacc.utexas.edu:/scratch/06414/tg857131/Magrathea/ /Volumes/stoneley5TB/all/Magrathea/v3
```

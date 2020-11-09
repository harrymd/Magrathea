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
```

where `tet_max_vol` is the maximum allowed size of the tetrahedral mesh element, in km<sup>3</sup> and `order` is the finite-element order of the elements (1 or 2, i.e. linear or quadratic).
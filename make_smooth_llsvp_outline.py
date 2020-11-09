import os

import matplotlib.pyplot as plt
import numpy as np
import pyproj
from scipy import optimize
from scipy.signal import savgol_filter
from shapely.geometry.polygon import Polygon

def find_xy_in_centroid_coords(lon, lat):

    # Define the coordinate system of (lon, lat) on a sphere.
    proj_lonlat = pyproj.Proj(proj = 'longlat', ellps = 'sphere')

    # Use the mean longitude and latitude as the initial guess for the centroid.
    lon_c = np.mean(lon)
    lat_c = np.mean(lat)

    # Use the initial guess for the centroid to define an azimuthal equidistant coordinate system.
    proj_aeqd = pyproj.Proj(proj='aeqd', ellps = 'sphere', lon_0 = lon_c, lat_0 = lat_c)

    # Find the projected coordinates of the initial centroid.
    x_c, y_c = pyproj.transform(proj_lonlat, proj_aeqd, lon_c, lat_c)

    # Iteratively find the centroid.
    # diff_thresh   Convergence criterion: how close should successive guesses be to trigger termination.
    # i_max         Maximum number of iteratiors if convergence not achieved.
    diff_thresh = 1.0E-3
    i = 0
    i_max = 100
    diff = np.inf
    print('Iterative solution for centroid:')
    while (diff > diff_thresh) or (i + 1 >= i_max):

        # Get the transformed coordinates using the current projection.
        x, y = pyproj.transform(proj_lonlat, proj_aeqd, lon, lat)

        # Find the centroid of the polygon using the current projection.
        xy_shapely_tuple = ((xi, yi) for xi, yi in zip(x, y))
        polygon = Polygon(xy_shapely_tuple)
        x_c_new = polygon.centroid.x
        y_c_new = polygon.centroid.y
        
        # Calculate the difference between successive estimates.
        diff = np.sqrt((x_c_new - x_c)**2.0 + (y_c_new - y_c)**2.0)
        print('Iteration {:>3d}, change: {:>10.7e}'.format(i, diff))
        
        # Update.
        i = i + 1
        x_c = x_c_new
        y_c = y_c_new
        lon_c, lat_c = pyproj.transform(proj_aeqd, proj_lonlat, x_c, y_c)
        proj_aeqd = pyproj.Proj(proj='aeqd', ellps = 'sphere', lon_0 = lon_c, lat_0 = lat_c)

    # Do a final projection step with the new centroid.
    x, y = pyproj.transform(proj_lonlat, proj_aeqd, lon, lat)

    # Find the centroid of the polygon using the current projection.
    xy_shapely_tuple = ((xi, yi) for xi, yi in zip(x, y))
    polygon = Polygon(xy_shapely_tuple)
    x_c = polygon.centroid.x
    y_c = polygon.centroid.y
    lon_c, lat_c = pyproj.transform(proj_aeqd, proj_lonlat, x_c, y_c)

    return x, y, lon_c, lat_c

def smooth_along_path_length(s_uniform, s, x):

    # Interpolate the function onto a regular grid.
    x_uniform = np.interp(s_uniform, s, x)

    # Filter the function using the Savitzky-Golay method.
    order_filter = 2
    n_s = len(s_uniform)
    len_filter = int(round(n_s/10.0))
    if (len_filter % 2) == 0:
        len_filter = len_filter + 1
    #
    x_filter = savgol_filter(x_uniform, len_filter, order_filter, mode = 'wrap')

    # Slightly adjust the end values so there is no mis-tie at the end.
    len_end_fix = len_filter
    x_diff = x_filter[-1] - x_filter[0]
    d_x_end_fix = np.linspace(0.0, 0.5*x_diff, num = len_end_fix)
    x_filter[:len_end_fix] = x_filter[:len_end_fix] + d_x_end_fix[::-1]
    x_filter[-len_end_fix:] = x_filter[-len_end_fix:] - d_x_end_fix

    return x_filter

def main():
    
    # Define input directory.
    dir_input = 'input/llsvp/'

    # Load the longitude and latitude (in degrees) of the shape.
    path_llsvp = os.path.join(dir_input, 'llsvp_african_outline.txt')
    lon, lat = np.loadtxt(path_llsvp).T

    # Find the coordinates of the outline in a centred azimuthal equidistant projection.
    x, y, lon_c, lat_c = find_xy_in_centroid_coords(lon, lat)

    # Find the path length along the outline.
    d_s = np.sqrt((x[1:] - x[:-1])**2.0 + (y[1:] - y[:-1])**2.0)
    s = np.cumsum(d_s)
    s = np.insert(s, 0, 0.0)
    
    # Resample the path length on a fine uniform grid.
    n_points = len(x)
    s_max = np.max(s)
    n_s = 10*n_points 
    s_uniform = np.linspace(0.0, s_max, num = n_s)

    # Smooth the x and y curves separately.
    x_filter = smooth_along_path_length(s_uniform, s, x)
    y_filter = smooth_along_path_length(s_uniform, s, y)

    # Re-calculate the path length.
    d_s_filter = np.sqrt((x_filter[1:] - x_filter[:-1])**2.0 + (y_filter[1:] - y_filter[:-1])**2.0)
    s_filter = np.cumsum(d_s_filter)
    s_filter = np.insert(s_filter, 0, 0.0)

    # Convert back to lon/lat coordinates.
    proj_lonlat = pyproj.Proj(proj = 'longlat', ellps = 'sphere')
    proj_aeqd = pyproj.Proj(proj='aeqd', ellps = 'sphere', lon_0 = lon_c, lat_0 = lat_c)
    lon_filter, lat_filter = pyproj.transform(proj_aeqd, proj_lonlat, x_filter, y_filter)

    # Save.
    header = '{:>16.12f} {:>16.12f}'.format(lon_c, lat_c)
    out_array = np.array([s_filter, x_filter, y_filter, lon_filter, lat_filter]).T
    file_out = 'llsvp_smooth.txt'
    path_out = os.path.join(dir_input, file_out)
    print('Saving to {:}'.format(path_out))
    np.savetxt(path_out, out_array, header = header)

    return

def old():

    # Down-sample.
    n_new = n_points
    s_max = np.max(s_filter)
    s_new = np.linspace(0.0, s_max, n_new + 1)[:-1] 
    x_new = np.interp(s_new, s_filter, x_filter)
    y_new = np.interp(s_new, s_filter, y_filter)

    # Generate evenly-spaced points around the curve.
    with open('input.txt', 'r') as in_id:

        tet_max_vol = float(in_id.readline())

    l_min = 0.5*(2.0**(1.0/2.0))*(3.0**(1.0/3.0))*(tet_max_vol**(1.0/3.0))
    s_grid_appx = l_min*1.0E3 # Convert to m.
    n_s_sample = int(round(s_max/s_grid_appx))
    s_sample_uniform = np.linspace(0.0, s_max, num = n_s_sample + 1)[:-1]
    x_sample_uniform = np.interp(s_sample_uniform, s_filter, x_filter)
    y_sample_uniform = np.interp(s_sample_uniform, s_filter, y_filter)

    # Convert back to lon/lat coordinates.
    proj_lonlat = pyproj.Proj(proj = 'longlat', ellps = 'sphere')
    proj_aeqd = pyproj.Proj(proj='aeqd', ellps = 'sphere', lon_0 = lon_c, lat_0 = lat_c)
    lon_sample_uniform, lat_sample_uniform = pyproj.transform(proj_aeqd, proj_lonlat, x_sample_uniform, y_sample_uniform)

    # Save.
    out_array = np.array([lon_sample_uniform, lat_sample_uniform]).T
    file_out = 'lonlat_llsvp_{:>012.6e}.txt'.format(l_min)
    print('Saving to {:}'.format(file_out))
    np.savetxt(file_out, out_array)

    return

if __name__ == '__main__':

    main()

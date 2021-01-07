import os
import sys

#import matplotlib.pyplot as plt
import meshio
import numpy as np
import pygmsh
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon

# Some issues importing evtk modules, so do it like this:
import evtk
unstructuredGridToVTK = evtk.hl.unstructuredGridToVTK
VtkTriangle = evtk.vtk.VtkTriangle
VtkTetra = evtk.vtk.VtkTetra
VtkQuad = evtk.vtk.VtkQuad
VtkQuadraticTetra = evtk.vtk.VtkQuadraticTetra

# Utilities. ------------------------------------------------------------------
def LegendrePoly2(x):

    f = ((3.0*(x**2.0)) - 1.0)/2.0

    return f

def mkdir_if_not_exist(dir_):

    if not os.path.exists(dir_):

        os.mkdir(dir_)

    return

def RLonLat_to_XYZ(r, lon, lat):
    '''
    Converts from radius, longitude and latitude to Cartesian coordinates.
    '''

    theta = (np.pi/2.0) - lat
        
    x = r*np.sin(theta)*np.cos(lon)
    y = r*np.sin(theta)*np.sin(lon)
    z = r*np.cos(theta)
    
    return x, y, z

def XYZ_to_RLonLat(x, y, z):
    '''
    Converts from Cartesian coordinates to radius, longitude and latitude.
    '''

    r       = np.sqrt(x**2.0 + y**2.0 + z**2.0)
    theta   = np.arccos(z/r)
    lat     = (np.pi/2.0) - theta
    lon     = np.arctan2(y, x)

    return r, lon, lat

def RLonLatEll_to_XYZ(r, lon, lat, ell):
    '''
    Converts from radius, longitude and latitude to Cartesian coordinates.
    '''

    # Polar angle.
    theta = (np.pi/2.0) - lat

    # Adjust radius for ellipticity.
    # Dahlen and Tromp (1998) eq. 14.4.
    cos_theta = np.cos(theta)
    r_p = r*(1.0 - ((2.0/3.0)*ell*LegendrePoly2(cos_theta))) 

    # Points are moved radially outwards, so formula is otherwise unchanged.
    x, y, z = RLonLat_to_XYZ(r_p, lon, lat)
    
    return x, y, z

def get_real_points(mesh):
    '''
    The gmsh library adds extra points to help in the meshing process.
    These are returned by generate_mesh().
    It is not clear from the pygmsh documentation how to remove these points.
    Here we remove them by selecting only the points belonging to triangles.
    It is also necessary to update the triangulation indices.
    '''

    # Get a list of the indices of points used in the triangulation.
    tri = mesh.get_cells_type('triangle')
    i_mesh = np.sort(np.unique(tri.flatten()))

    # Get the points used in the triangulation.
    pts = mesh.points[i_mesh, :]
    #print('get_real_points', pts.shape)

    # Define a mapping such that tri_new[i, j] = i_mapping[tri[i, j].
    # Note that i_mapping[k] == -1 if k does not belong to i_mesh.
    i_max = int(np.max(i_mesh))
    i_mapping = np.zeros((i_max + 1), dtype = np.int) - 1
    j = 0
    for i in range(i_max + 1):

        if i in i_mesh:

            i_mapping[i] = j

            j = j + 1

    # Apply the mapping to create a new triangulation.
    n_tri = tri.shape[0]
    tri_new = np.zeros(tri.shape, dtype = np.int) - 1
    for i in range(n_tri):

        for j in range(3):

            tri_new[i, j] = i_mapping[tri[i, j]] 

    return pts, tri_new, i_mapping

def XYZ_to_REll(x, y, z, r_ellipticity_profile, ellipticity_profile):

    r_oblate = np.sqrt((x**2.0) + (y**2.0) + (z**2.0))
    #r_h_oblate = np.sqrt((x**2.0) + (y**2.0))
    #theta = np.arctan2(z, r_h_oblate) # Note: Sign of theta not important.
    cos_theta = z/r_oblate # Note sign of theta is not important.
    P2CosTheta = LegendrePoly2(cos_theta)

    # First guess of ellipticity, based on oblate radius.
    # (Points outside the sphere will have the ellipticity of the outer surface.)
    r_spherical_estimate_previous = r_oblate.copy()
    ellipticity_estimate_previous = np.interp(r_oblate, r_ellipticity_profile, ellipticity_profile)
    #
    r_spherical_estimate_new = np.zeros(r_oblate.shape)
    ellipticity_estimate_new = np.zeros(r_oblate.shape)
    change_r_spherical_estimate = np.zeros(r_oblate.shape)

    # Iteratively solve.
    print('XYZ_to_REll()')

    n_iterations_max = 100
    n_pts = x.size
    converged = np.zeros(x.shape, dtype = np.bool)
    # Target convergence between successive iterations (km).
    thresh = 1.0E-10
    successful = False # Flag for successful convergence.
    for i in range(n_iterations_max):

        #j_not_converged = np.where(~converged)[0]
        j_not_converged = ~converged
        k = np.unravel_index(j_not_converged.argmax(), j_not_converged.shape)

        # Estimate spherical radius from previous estimate of ellipticity.
        r_spherical_estimate_new[j_not_converged] = r_oblate[j_not_converged]/(1.0 - ((2.0/3.0)*ellipticity_estimate_previous[j_not_converged]*P2CosTheta[j_not_converged]))
        ellipticity_estimate_new[j_not_converged] = np.interp(r_spherical_estimate_new[j_not_converged], r_ellipticity_profile, ellipticity_profile)

        # Check for convergence.
        change_r_spherical_estimate[j_not_converged] = np.abs(r_spherical_estimate_new[j_not_converged] - r_spherical_estimate_previous[j_not_converged])

        #converged_copy = converged.copy()
        converged[j_not_converged] = (change_r_spherical_estimate[j_not_converged] < thresh)
        n_converged = np.sum(converged)

        print('Iteration {:>5d} (max.: {:>5d}), {:>9d} out of {:>9d} points converged.'.format(i + 1, n_iterations_max, n_converged, n_pts))

        # Case 1: All values have converged.
        if n_converged == n_pts:
            
            successful = True
            break

        # Case 2: Some values have not converged.
        else:

            # Prepare for next iteration.
            r_spherical_estimate_previous[j_not_converged] = r_spherical_estimate_new[j_not_converged]
            ellipticity_estimate_previous[j_not_converged] = ellipticity_estimate_new[j_not_converged]

    if not successful:
        
        #print('These points did not converge.')
        #print(x[j_not_converged])
        #print(y[j_not_converged])
        #print(z[j_not_converged])
        raise ValueError('Iterative solution for r did not converge.')

    return r_spherical_estimate_new, ellipticity_estimate_new

def test_XYZ_to_REll(case):

    # Case 1: Random points.
    # Case 2: Specific difficult points.

    if case == 1:

        # Generate random points inside sphere.
        r_max = 6371.0
        n_pts = 1000
        x_spherical = np.zeros(n_pts)
        y_spherical = np.zeros(n_pts)
        z_spherical = np.zeros(n_pts)
        i = 0
        while i < n_pts:

            x_random = np.random.uniform(low = -r_max, high = r_max)
            y_random = np.random.uniform(low = -r_max, high = r_max)
            z_random = np.random.uniform(low = -r_max, high = r_max)

            r_random = np.sqrt((x_random**2.0) + (y_random**2.0) + (z_random**2.0))

            if r_random <= r_max:

                x_spherical[i] = x_random
                y_spherical[i] = y_random
                z_spherical[i] = z_random

                i = i + 1

    elif case == 2:

        x_spherical = np.array([-5175.87774264, -1281.64533839])
        y_spherical = np.array([ 2036.66020355,  5676.29719589])
        z_spherical = np.array([  384.22039391, -1240.90543724])

    # Get spherical coordinates.
    r_spherical, lon, lat = XYZ_to_RLonLat(x_spherical, y_spherical, z_spherical) 

    # Load ellipticity profile.
    ellipticity_data = np.loadtxt('../../input/Magrathea/llsvp/ellipticity_profile.txt')
    r_ellipticity_profile, ellipticity_profile = ellipticity_data.T
    r_ellipticity_profile = r_ellipticity_profile*1.0E-3 # Convert from m to km.

    # Interpolate ellipticity profile at points.
    ellipticity = np.interp(r_spherical, r_ellipticity_profile, ellipticity_profile)

    # Distort spherical coordinates according to ellipticity.
    theta = (np.pi/2.0) - lat
    cos_theta = np.cos(theta)
    P2CosTheta = LegendrePoly2(cos_theta)
    r_factor = (1.0 - ((2.0/3.0)*ellipticity*P2CosTheta))
    r_oblate = r_spherical*r_factor

    # Find the x, y, z coordinates of the distorted points.
    x_oblate, y_oblate, z_oblate = RLonLat_to_XYZ(r_oblate, lon, lat)

    # Try to convert back to r_spherical.
    r_spherical_estimate, ellipticity_estimate = XYZ_to_REll(x_oblate, y_oblate, z_oblate, r_ellipticity_profile, ellipticity_profile)

    return

# Building the mesh. ----------------------------------------------------------
def make_ellipsoidal_poly_file_wrapper(dir_input, subdir_out, tet_max_vol, model, discon_lists, name, ellipticity_data, mesh_size_maxima):

    # Define output path.
    path_poly = os.path.join(subdir_out, '{:}.poly'.format(name))
    if os.path.exists(path_poly):

        print('Tetgen .poly input file already exists, skipping.')
        return path_poly

    # Determine the mesh size on the CMB sphere. 
    mesh_size = 0.5*(2.0**(1.0/2.0))*(3.0**(1.0/3.0))*(tet_max_vol**(1.0/3.0))
    #mesh_size = 3.0*mesh_size

    r_discons = model['r'][discon_lists['all']]
    n_discons = len(r_discons)

    mesh_size_list = np.zeros(n_discons) + mesh_size
    i_mesh_too_big = (mesh_size_list > mesh_size_maxima)
    mesh_size_list[i_mesh_too_big] = mesh_size_maxima[i_mesh_too_big]

    # Load ellipticity data.
    if ellipticity_data is None:

        raise NotImplementedError('Extra discontinuities in mantle not implemented for non-ellipsoidal models.')

        ellipticity_srf = None
        ellipticity_cmb = None
        ellipticity_icb = None

    else:

        r_ellipticity_profile = ellipticity_data[:, 0]
        r_ellipticity_profile = r_ellipticity_profile*1.0E-3 # Convert to km.
        ellipticity_profile = ellipticity_data[:, 1]

        ellipticity    = np.interp( r_discons,
                                    r_ellipticity_profile,
                                    ellipticity_profile)

    # Set output file for mesh points. 
    out_path_i_pts_inside_llsvp_on_cmb = os.path.join(subdir_out, 'i_pts_inside_llsvp_on_cmb.npy')
    out_path_i_pts_boundary_llsvp_on_cmb = os.path.join(subdir_out, 'i_pts_boundary_llsvp_on_cmb.npy')

    out_paths_pts = []
    out_paths_tri = []
    for i in range(discon_lists['n_discons']):

        out_path_pts_discon_i = os.path.join(subdir_out, 'discon_{:>1d}_pts.npy'.format(i))
        out_path_tri_discon_i = os.path.join(subdir_out, 'discon_{:>1d}_tri.npy'.format(i))

        out_paths_pts.append(out_path_pts_discon_i)
        out_paths_tri.append(out_path_tri_discon_i)

    # Load mesh points if already exist.
    path_list = [   *out_paths_pts, *out_paths_tri,
                    out_path_i_pts_inside_llsvp_on_cmb,
                    out_path_i_pts_boundary_llsvp_on_cmb]
    files_exist = all([os.path.exists(path_) for path_ in path_list])

    pts_list = []
    tri_list = []

    if files_exist: 

        print('Points and triangulation files already exist.')
        for i in range(discon_lists['n_discons']):

            print('Loading {:}'.format(out_paths_pts[i]))
            pts_list.append(np.load(out_paths_pts[i]))

            print('Loading {:}'.format(out_paths_tri[i]))
            tri_list.append(np.load(out_paths_tri[i]))

        print('Loading {:}.'.format(out_path_i_pts_inside_llsvp_on_cmb))
        i_pts_inside_llsvp_on_cmb = np.load(out_path_i_pts_inside_llsvp_on_cmb)
        print('Loading {:}.'.format(out_path_i_pts_boundary_llsvp_on_cmb))
        i_pts_boundary_llsvp_on_cmb = np.load(out_path_i_pts_boundary_llsvp_on_cmb)

    # Otherwise, create mesh points.
    else:

        for i in range(discon_lists['n_discons']):
            
            r_discon = model['r'][discon_lists['all'][i]]
            name_i = 'discon_{:>1d}'.format(i)

            if (i == discon_lists['j_cmb']):
                
                pts_i, tri_i,                                               \
                i_pts_inside_llsvp_on_cmb, i_pts_boundary_llsvp_on_cmb  =   \
                build_cmb_ellipsoid(subdir_out, dir_input, r_discon, mesh_size_list[i], ellipticity[i], name_i)

            else:

                pts_i, tri_i = build_ellipsoid(subdir_out, r_discon, mesh_size_list[i], ellipticity[i], name_i)

            pts_list.append(pts_i)
            tri_list.append(tri_i)

    # Add the points defining the top surface of the LLSVP.
    # These are simply copies of the points in the bottom surface of the
    # LLSVP, moved outwards by the height of the LLSVP.
    d_r_llsvp = 400.0 # Height of LLSVP in km.
    #
    # First, find all of the points in the bottom surface of the LLSVP.
    #i_pts_llsvp_on_cmb = np.concatenate([i_pts_inside_llsvp_on_cmb,
    #                        i_pts_boundary_llsvp_on_cmb]).astype(np.int)
    i_pts_llsvp_on_cmb = np.concatenate([i_pts_boundary_llsvp_on_cmb,
                                         i_pts_inside_llsvp_on_cmb]).astype(np.int)

    # Create a mapping between the triangulation indices and a list of new
    # points.
    n_pts_llsvp_top = len(i_pts_llsvp_on_cmb)
    i_max_index_llsvp_on_cmb = np.max(i_pts_llsvp_on_cmb)
    mapping_i_llsvp_base_i_llsvp_top = np.zeros(i_max_index_llsvp_on_cmb + 1, dtype = np.int) - 1
    #
    for j, i in enumerate(i_pts_llsvp_on_cmb):

        mapping_i_llsvp_base_i_llsvp_top[i] = j

    # Create the new points on the top surface of the LLSVP.
    i_llsvp_top = np.array(list(range(n_pts_llsvp_top)), dtype = np.int)
    pts_cmb = pts_list[discon_lists['j_cmb']]
    pts_llsvp_on_cmb = pts_cmb[:, i_pts_llsvp_on_cmb]
    r_cmb = r_discons[discon_lists['j_cmb']]
    if ellipticity_data is None:

        pts_llsvp_top = move_points_outwards_spherical(pts_llsvp_on_cmb, d_r_llsvp)

    else:
        
        ellipticity_top_llsvp = np.interp((r_cmb + d_r_llsvp),
                r_ellipticity_profile, ellipticity_profile)

        pts_llsvp_top = move_points_outwards_ellipsoidal(pts_llsvp_on_cmb, d_r_llsvp,
                r_cmb, ellipticity_top_llsvp)
        
    # Next, find all of the triangles in the bottom surface of the LLSVP.
    tri_cmb = tri_list[discon_lists['j_cmb']]
    n_tri_cmb = tri_cmb.shape[1]
    tri_cmb_in_llsvp = np.zeros(n_tri_cmb, dtype = np.bool)

    for i in range(n_tri_cmb):

        if np.all(np.isin(tri_cmb[:, i], i_pts_llsvp_on_cmb)):

            tri_cmb_in_llsvp[i] = True

    tri_llsvp_base = tri_cmb[:, tri_cmb_in_llsvp]
    n_tri_llsvp_base = tri_llsvp_base.shape[1]

    # Apply the mapping to find the triangulation on the top of the LLSVP.
    n_tri_llsvp_top = n_tri_llsvp_base
    tri_llsvp_top = np.zeros((3, n_tri_llsvp_top), dtype = np.int)
    for i in range(n_tri_llsvp_top):

        for j in range(3):

            index_new = mapping_i_llsvp_base_i_llsvp_top[tri_llsvp_base[j, i]]
            assert index_new > -1
            tri_llsvp_top[j, i] = index_new

    # Apply offsets.
    n_pts_cumulative = 0
    for i in range(discon_lists['n_discons']):

        tri_list[i] = tri_list[i] + n_pts_cumulative
        n_pts_i = pts_list[i].shape[1]

        if (i == discon_lists['j_cmb']):

            tri_llsvp_base = tri_llsvp_top + n_pts_cumulative
            i_pts_inside_llsvp_on_cmb = i_pts_inside_llsvp_on_cmb + n_pts_cumulative
            i_pts_boundary_llsvp_on_cmb = i_pts_boundary_llsvp_on_cmb + n_pts_cumulative
            i_pts_llsvp_on_cmb = i_pts_llsvp_on_cmb + n_pts_cumulative

        n_pts_cumulative = n_pts_cumulative + n_pts_i

    i_llsvp_top = i_llsvp_top + n_pts_cumulative
    tri_llsvp_top = tri_llsvp_top + n_pts_cumulative

    # Define the sides of the LLSVP as quadrilaterals.
    # Each quadrilateral joins one edge of the upper outline to one edge
    # of the lower outline.
    n_llsvp_outline_pts = len(i_pts_boundary_llsvp_on_cmb)
    n_llsvp_outline_edges = n_llsvp_outline_pts
    #
    quad_llsvp_sides = np.zeros((4, n_llsvp_outline_edges), dtype = np.int)
    for i in range(n_llsvp_outline_edges):

        i0 = i
        i1 = (i + 1) % n_llsvp_outline_edges

        i0_base = i_pts_boundary_llsvp_on_cmb[i0]
        i1_base = i_pts_boundary_llsvp_on_cmb[i1]
        i0_top  = i_llsvp_top[i0]
        i1_top  = i_llsvp_top[i1]

        quad_llsvp_sides[:, i] = [i0_base, i1_base, i1_top, i0_top]

    # Define the regions.
    # Note: ellipticity is not accounted for, because typical flattening
    # for Earth (~1/300) does not move the mid-points of each shell into
    # a different shell.
    # The polygonal surfaces define four regions (1 : inner core, 2 : outer core,
    # 3 : mantle, and 4 : LLSVP). These are identified in the .poly file by a single
    # point placed within them.
    # The points for the outer core and mantle are placed along the x-axis,
    r_discons_with_zero = np.insert(r_discons, 0, 0.0)
    x_shell_points = r_discons_with_zero[0:-1] + 0.5*np.diff(r_discons_with_zero)
    shell_points = np.array([[x_shell_points[i], 0.0, 0.0] for i in range(discon_lists['n_discons'])])
    #
    # The point for the LLSVP is placed at the centroid of the outline,
    # halfway between the inner and outer surfaces.
    path_llsvp = os.path.join(dir_input, 'llsvp_smooth.txt')
    with open(path_llsvp, 'r') as in_id:
        
        _, llsvp_centre_lon_deg, llsvp_centre_lat_deg = in_id.readline().split()

    llsvp_centre_lon = np.deg2rad(float(llsvp_centre_lon_deg))
    llsvp_centre_lat = np.deg2rad(float(llsvp_centre_lat_deg))
    llsvp_centre_r = r_cmb + (d_r_llsvp/2.0) 
    pt_centre_llsvp = list(RLonLat_to_XYZ(llsvp_centre_r, llsvp_centre_lon, llsvp_centre_lat))
    #
    pts_regions = np.array([*shell_points, pt_centre_llsvp]).T

    # Write the .poly file.
    make_ellipsoidal_poly_file(pts_list, pts_llsvp_top, tri_list, tri_llsvp_top, quad_llsvp_sides, pts_regions, path_poly)

    return path_poly

def build_ellipsoid(subdir_out, r, mesh_size, ellipticity, name = None):

    # Create the sphere mesh with embedded LLSVP outline.
    with pygmsh.geo.Geometry() as geom:

        # Create a representation of the ellipsoidal surface and add it 
        # to the geometry.
        spherical_shell = create_spherical_shell(geom, mesh_size, r)

        # Generate the mesh.
        mesh = geom.generate_mesh(dim = 2, verbose = False)

    # Remove construction points.
    pts, tri, i_mapping = get_real_points(mesh)
    pts = pts.T
    tri = tri.T

    # Rescale to ellipsoid.
    pts = rescale_to_ellipsoid(pts, r, ellipticity)

    if name is not None:

        # Second, save points.
        out_path_pts = os.path.join(subdir_out, '{:}_pts.npy'.format(name))
        print('Saving points to {:}'.format(out_path_pts))
        np.save(out_path_pts, pts)

        # Third, save triangulation.
        out_path_tri = os.path.join(subdir_out, '{:}_tri.npy'.format(name))
        print('Saving triangulation to {:}'.format(out_path_tri))
        np.save(out_path_tri, tri)

    return pts, tri

def build_ellipsoid_v0(subdir_out, r, mesh_size, ellipticity, name = None):

    # Create the sphere mesh with embedded LLSVP outline.
    with pygmsh.geo.Geometry() as geom:

        # Create a representation of the ellipsoidal surface and add it 
        # to the geometry.
        ellipsoid_shell = create_ellipsoid_shell(geom, mesh_size, r, ellipticity)

        # Generate the mesh.
        mesh = geom.generate_mesh(dim = 2, verbose = False)

    # Remove construction points.
    pts, tri, i_mapping = get_real_points(mesh)
    pts = pts.T
    tri = tri.T

    # Unfortunately, it seems that Pygmsh doesn't guarantee that the points
    # fall on the surface of the sphere/ellipsoid, so here we readjust them.
    pts = rescale_to_surface(pts, r, ellipticity)

    if name is not None:

        # Save points and triangulation.
        # First create output directory.
        #subdir_out = os.path.join(dir_output, 'sphere_{:>07.2f}'.format(mesh_size))
        #for dir_ in [dir_output, subdir_out]:
        #    mkdir_if_not_exist(dir_)
        #mkdir_if_not_exist(subdir_out)

        # Second, save points.
        out_path_pts = os.path.join(subdir_out, '{:}_pts.npy'.format(name))
        print('Saving points to {:}'.format(out_path_pts))
        np.save(out_path_pts, pts)

        # Third, save triangulation.
        out_path_tri = os.path.join(subdir_out, '{:}_tri.npy'.format(name))
        print('Saving triangulation to {:}'.format(out_path_tri))
        np.save(out_path_tri, tri)

    return pts, tri

def build_cmb_ellipsoid(subdir_out, dir_input, r_cmb, mesh_size, ellipticity_cmb, name, make_plots = False):

    # Define a mesh coarsening factor.
    # Seems necessary to coarsen the mesh near the LLSVP boundary to maintain
    # a relatively even mesh size.
    #mesh_size_factor = 2.0 
    mesh_size_factor = 1.0

    # Load the LLSVP outline and find the coordinates of the points with the
    # chosen spacing by linear interpolation.
    path_llsvp = os.path.join(dir_input, 'llsvp_smooth.txt')
    s_llsvp, _, _, lon_llsvp, lat_llsvp = np.loadtxt(path_llsvp).T
    s_llsvp = s_llsvp*1.0E-3 # Convert to km.
    s_max = np.max(s_llsvp)
    n_s_sample = int(round(s_max/(mesh_size_factor*mesh_size)))
    s_pts = np.linspace(0.0, s_max, num = n_s_sample + 1)[:-1]
    lon_pts = np.interp(s_pts, s_llsvp, lon_llsvp)
    lat_pts = np.interp(s_pts, s_llsvp, lat_llsvp)

    # Rotate the LLSVP, to avoid the grid lines used by gmsh during
    # construction.
    shift = 60.0
    lon_pts = lon_pts + shift 

    # Find the Cartesian coordinates of the points.
    lon_pts_rads = np.deg2rad(lon_pts)
    lat_pts_rads = np.deg2rad(lat_pts)
    x_pts_sph, y_pts_sph, z_pts_sph = RLonLat_to_XYZ(r_cmb, lon_pts_rads, lat_pts_rads) 
    pts_sph = np.array([x_pts_sph, y_pts_sph, z_pts_sph]).T
    n_pts = pts_sph.shape[0]

    # Create the ellipsoidal mesh with embedded LLSVP outline.
    with pygmsh.geo.Geometry() as geom:

        spherical_shell = create_spherical_shell(geom, mesh_size, r_cmb)

        # Add the points that form the point-wise linear boundary of the LLSVP.
        pt_list = []
        for i in range(n_pts):

            pt = geom.add_point(pts_sph[i, :], mesh_size = mesh_size_factor*mesh_size)
            pt_list.append(pt)

        # Add the straight-line segments that form the boundary of the LLSVP.
        line_list = []
        for i in range(n_pts):

            i0 = i
            i1 = (i + 1) % n_pts

            line = geom.add_line(pt_list[i0], pt_list[i1])
            geom.in_surface(line, spherical_shell.surface_loop)
            line_list.append(line)

        curve_loop = geom.add_curve_loop(line_list)

        # Generate the mesh.
        mesh = geom.generate_mesh(dim = 2, verbose = False)

    # Find the indices of the points in the outline.
    # The first 7 vertices are construction vertices which are ignored.
    vertices = mesh.get_cells_type('vertex')
    i_outline = np.squeeze(vertices[7:])

    # Remove construction points.
    pts, tri, i_mapping = get_real_points(mesh)
    pts = pts.T
    tri = tri.T

    # Rescale to ellipsoid.
    pts = rescale_to_ellipsoid(pts, r_cmb, ellipticity_cmb)

    # Re-label the outline points after construction points have been removed.
    i_outline_new = np.zeros(i_outline.shape, i_outline.dtype)
    for j, i in enumerate(i_outline):
        i_outline_new[j] = i_mapping[i]
    i_outline = i_outline_new

    # Find the points belonging to the LLSVP.
    # First, create Shapely polygon representing the points.
    x, y, z = pts
    _, lon, lat = XYZ_to_RLonLat(x, y, z)
    lon_deg = np.rad2deg(lon)
    lat_deg = np.rad2deg(lat)
    pts_deg_array = np.squeeze(np.array([lon_deg[i_outline], lat_deg[i_outline]]))
    polygon = Polygon(pts_deg_array.T)
    # Second, check each point to see if it is inside the polygon.
    i_inside = []
    n_pts = len(x)
    for i in range(n_pts):

        pt = Point((lon_deg[i], lat_deg[i]))

        if polygon.contains(pt):

            i_inside.append(i)
    
    i_inside = np.array(i_inside, dtype = np.int)

    # Remove longitude shift.
    lon_deg = lon_deg - shift
    i_2pi_shift = (lon_deg < -180.0)
    lon_deg[i_2pi_shift] = lon_deg[i_2pi_shift] + 360.0

    # Convert back to x, y and z.
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    x, y, z = RLonLatEll_to_XYZ(r_cmb, lon_rad, lat_rad, ellipticity_cmb)

    if name is not None:

        # Second, save points.
        pts = np.array([x, y, z])
        out_path_pts = os.path.join(subdir_out, '{:}_pts.npy'.format(name))
        print('Saving points to {:}'.format(out_path_pts))
        np.save(out_path_pts, pts)

        # Third, save triangulation.
        out_path_tri = os.path.join(subdir_out, '{:}_tri.npy'.format(name))
        print('Saving triangulation to {:}'.format(out_path_tri))
        np.save(out_path_tri, tri)

        # Fourth, save list of inside points.
        out_path_i_pts_inside = os.path.join(subdir_out, 'i_pts_inside_llsvp_on_cmb.npy')
        print('Saving list of indices of LLSVP interior points to {:}'.format(out_path_i_pts_inside)) 
        np.save(out_path_i_pts_inside, i_inside)

        # Fifth, save list of points on boundary of LLSVP.
        out_path_i_pts_boundary = os.path.join(subdir_out, 'i_pts_boundary_llsvp_on_cmb.npy')
        print('Saving list of indices of LLSVP boundary points to {:}'.format(out_path_i_pts_boundary)) 
        np.save(out_path_i_pts_boundary, i_outline)

    make_plots = False
    if make_plots:

        import matplotlib.pyplot as plt

        print(ellipticity_cmb)
        print(1.0/ellipticity_cmb)

        # Plot points in cross-section.
        fig, ax_arr = plt.subplots(1, 2, figsize = (11.0, 8.5))
        ax  = ax_arr[0] 
        
        sign_y = np.sign(y)
        sign_y[sign_y == 0.0] = 1.0
        r_h = np.sqrt(x**2.0 + y**2.0)*sign_y

        theta_span = np.linspace(0.0, 2.0*np.pi, num = 1000)
        r_h_circle = r_cmb*np.sin(theta_span)
        z_circle = r_cmb*np.cos(theta_span)

        cos_theta_span = np.cos(theta_span)
        r_ellipse = r_cmb*(1.0 - (2.0/3.0)*ellipticity_cmb*LegendrePoly2(cos_theta_span))
        r_h_ellipse = r_ellipse*np.sin(theta_span)
        z_ellipse = r_ellipse*np.cos(theta_span)

        ax.scatter(r_h, z, s = 3, c = 'k', zorder = 10)

        ax.plot(r_h_circle, z_circle, c = 'r', label = 'Sphere')
        ax.plot(r_h_ellipse, z_ellipse, c = 'b', label = 'Ellipsoid')

        ax.legend()

        ax.set_aspect(1.0)

        ax = ax_arr[1]

        r_pts = np.sqrt((x**2.0 + y**2.0 + z**2.0))
        r_h = np.sqrt(x**2.0 + y**2.0)
        
        theta_pts = np.arctan2(r_h, z)
        cos_theta_pts = np.cos(theta_pts)
        
        r_from_theta_pts = r_cmb*(1.0 - ((2.0/3.0)*ellipticity_cmb*LegendrePoly2(cos_theta_pts)))

        max_r_error = np.max(np.abs(r_from_theta_pts - r_pts))
        print('Maximum r error = {:>.3e} (units of R_cmb)'.format(max_r_error/r_cmb))

        ax.scatter(r_pts - r_cmb, r_from_theta_pts - r_cmb, s = 3, c = 'k')
        ax.plot([-15.0, 0.0], [-15.0, 0.0])

        ax.set_xlabel('Radii of points - CMB radius')
        ax.set_ylabel('Expected radii at point latitude - CMB radius')

        ax.set_aspect(1.0)
        
        # -----------------------------------------------------------------

        # Plot points on 2D map.
        fig = plt.figure(figsize = (11.0, 8.5))
        ax = plt.gca()

        i_east = np.where(lon > 0.0)[0]
        n_tri = tri.shape[1]
        mask = np.zeros(n_tri, dtype = np.bool)
        mask[:] = True
        for i in range(n_tri):

            lon_deg_i = lon_deg[tri[:, i]]
            lon_diffs_i = [lon_deg_i[0] - lon_deg_i[1], lon_deg_i[0] - lon_deg_i[2],
                            lon_deg_i[1] - lon_deg_i[2]]
            max_lon_diff = np.max(np.abs(lon_diffs_i))

            if max_lon_diff < 180.0:

                mask[i] = False

        ax.triplot(lon_deg, lat_deg, triangles = tri.T, mask = mask)
        #ax.scatter(lon_deg[i_outline], lat_deg[i_outline], c = 'r', zorder = 10)
        ax.plot(lon_deg[i_outline], lat_deg[i_outline], c = 'r', marker = '.', linestyle = '-', zorder = 10)
        ax.scatter(lon_deg[i_inside], lat_deg[i_inside], c = 'g', zorder = 11)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xlim([-180.0, 180.0])
        ax.set_ylim([-90.0, 90.0])
        
        # Plot points in 3D.
        fig = plt.figure(figsize = (8.5, 8.5))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_trisurf(x, y, z, triangles = tri.T, edgecolor = (1.0, 0.0, 0.0, 0.2), color = (1.0, 1.0, 1.0, 1.0)) #cmap = 'magma') 
        ax.plot(x[i_outline], y[i_outline], z[i_outline], zorder = 10)
        ax.scatter(x[i_inside], y[i_inside], z[i_inside], color = 'g', zorder = 11)

        plt.show()

    return pts, tri, i_inside, i_outline

def build_cmb_ellipsoid_v0(subdir_out, dir_input, r_cmb, mesh_size, ellipticity_cmb, name, make_plots = False):

    # Define a mesh coarsening factor.
    # Seems necessary to coarsen the mesh near the LLSVP boundary to maintain
    # a relatively even mesh size.
    #mesh_size_factor = 2.0 
    mesh_size_factor = 1.0

    # Load the LLSVP outline and find the coordinates of the points with the
    # chosen spacing by linear interpolation.
    path_llsvp = os.path.join(dir_input, 'llsvp_smooth.txt')
    s_llsvp, _, _, lon_llsvp, lat_llsvp = np.loadtxt(path_llsvp).T
    s_llsvp = s_llsvp*1.0E-3 # Convert to km.
    s_max = np.max(s_llsvp)
    n_s_sample = int(round(s_max/(mesh_size_factor*mesh_size)))
    s_pts = np.linspace(0.0, s_max, num = n_s_sample + 1)[:-1]
    lon_pts = np.interp(s_pts, s_llsvp, lon_llsvp)
    lat_pts = np.interp(s_pts, s_llsvp, lat_llsvp)

    # Rotate the LLSVP, to avoid the grid lines used by gmsh during
    # construction.
    shift = 60.0
    lon_pts = lon_pts + shift 

    # Find the Cartesian coordinates of the points.
    lon_pts_rads = np.deg2rad(lon_pts)
    lat_pts_rads = np.deg2rad(lat_pts)
    if ellipticity_cmb is None:

        x_pts, y_pts, z_pts = RLonLat_to_XYZ(r_cmb, lon_pts_rads, lat_pts_rads) 

    else:

        x_pts, y_pts, z_pts = RLonLatEll_to_XYZ(r_cmb, lon_pts_rads, lat_pts_rads, ellipticity_cmb) 

    pts = np.array([x_pts, y_pts, z_pts]).T
    n_pts = pts.shape[0]

    # Create the ellipsoidal mesh with embedded LLSVP outline.
    with pygmsh.geo.Geometry() as geom:

        ellipsoid_shell = create_ellipsoid_shell(geom, mesh_size, r_cmb, ellipticity_cmb)

        # Add the points that form the point-wise linear boundary of the LLSVP.
        pt_list = []
        for i in range(n_pts):

            pt = geom.add_point(pts[i, :], mesh_size = mesh_size_factor*mesh_size)
            pt_list.append(pt)

        # Add the straight-line segments that form the boundary of the LLSVP.
        line_list = []
        for i in range(n_pts):

            i0 = i
            i1 = (i + 1) % n_pts

            line = geom.add_line(pt_list[i0], pt_list[i1])
            geom.in_surface(line, ellipsoid_shell.surface_loop)
            line_list.append(line)

        curve_loop = geom.add_curve_loop(line_list)

        # Generate the mesh.
        mesh = geom.generate_mesh(dim = 2, verbose = False)

    # Find the indices of the points in the outline.
    # The first 7 vertices are construction vertices which are ignored.
    vertices = mesh.get_cells_type('vertex')
    i_outline = np.squeeze(vertices[7:])

    # Remove construction points.
    pts, tri, i_mapping = get_real_points(mesh)
    tri = tri.T

    # Unfortunately, it seems that Pygmsh doesn't guarantee that the points
    # fall on the surface of the sphere/ellipsoid, so here we readjust them.
    pts = rescale_to_surface(pts, r_cmb, ellipticity_cmb)

    # Re-label the outline points after construction points have been removed.
    i_outline_new = np.zeros(i_outline.shape, i_outline.dtype)
    for j, i in enumerate(i_outline):
        i_outline_new[j] = i_mapping[i]
    i_outline = i_outline_new

    ## Get the points and the triangulation.
    #pts = mesh.points
    #tri = mesh.get_cells_type('triangle')

    # Find the points belonging to the LLSVP.
    # First, create Shapely polygon representing the points.
    x, y, z = pts.T
    _, lon, lat = XYZ_to_RLonLat(x, y, z)
    lon_deg = np.rad2deg(lon)
    lat_deg = np.rad2deg(lat)
    pts_deg_array = np.squeeze(np.array([lon_deg[i_outline], lat_deg[i_outline]]))
    polygon = Polygon(pts_deg_array.T)
    # Second, check each point to see if it is inside the polygon.
    i_inside = []
    n_pts = len(pts)
    for i in range(n_pts):

        pt = Point((lon_deg[i], lat_deg[i]))

        if polygon.contains(pt):

            i_inside.append(i)

    i_inside = np.array(i_inside, dtype = np.int)

    # Remove longitude shift.
    lon_deg = lon_deg - shift
    i_2pi_shift = (lon_deg < -180.0)
    lon_deg[i_2pi_shift] = lon_deg[i_2pi_shift] + 360.0

    # Convert back to x, y and z.
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    x, y, z = RLonLatEll_to_XYZ(r_cmb, lon_rad, lat_rad, ellipticity_cmb)

    if name is not None:

        # Save points and triangulation.
        # First create output directory.
        #subdir_out = os.path.join(dir_output, 'sphere_{:>07.2f}'.format(mesh_size))
        #for dir_ in [dir_output, subdir_out]:
        #    mkdir_if_not_exist(dir_)
        #mkdir_if_not_exist(subdir_out)

        # Second, save points.
        pts = np.array([x, y, z])
        out_path_pts = os.path.join(subdir_out, '{:}_pts.npy'.format(name))
        print('Saving points to {:}'.format(out_path_pts))
        np.save(out_path_pts, pts)

        # Third, save triangulation.
        out_path_tri = os.path.join(subdir_out, '{:}_tri.npy'.format(name))
        print('Saving triangulation to {:}'.format(out_path_tri))
        np.save(out_path_tri, tri)

        # Fourth, save list of inside points.
        out_path_i_pts_inside = os.path.join(subdir_out, 'i_pts_inside_llsvp_on_cmb.npy')
        print('Saving list of indices of LLSVP interior points to {:}'.format(out_path_i_pts_inside)) 
        np.save(out_path_i_pts_inside, i_inside)

        # Fifth, save list of points on boundary of LLSVP.
        out_path_i_pts_boundary = os.path.join(subdir_out, 'i_pts_boundary_llsvp_on_cmb.npy')
        print('Saving list of indices of LLSVP boundary points to {:}'.format(out_path_i_pts_boundary)) 
        np.save(out_path_i_pts_boundary, i_outline)

    #make_plots = False 
    if make_plots:

        import matplotlib.pyplot as plt

        print(ellipticity_cmb)
        print(1.0/ellipticity_cmb)

        # Plot points in cross-section.
        fig, ax_arr = plt.subplots(1, 2, figsize = (11.0, 8.5))
        ax  = ax_arr[0] 
        
        sign_y = np.sign(y)
        sign_y[sign_y == 0.0] = 1.0
        r_h = np.sqrt(x**2.0 + y**2.0)*sign_y

        theta_span = np.linspace(0.0, 2.0*np.pi, num = 1000)
        r_h_circle = r_cmb*np.sin(theta_span)
        z_circle = r_cmb*np.cos(theta_span)

        cos_theta_span = np.cos(theta_span)
        r_ellipse = r_cmb*(1.0 - (2.0/3.0)*ellipticity_cmb*LegendrePoly2(cos_theta_span))
        r_h_ellipse = r_ellipse*np.sin(theta_span)
        z_ellipse = r_ellipse*np.cos(theta_span)

        ax.scatter(r_h, z, s = 3, c = 'k', zorder = 10)

        ax.plot(r_h_circle, z_circle, c = 'r', label = 'Sphere')
        ax.plot(r_h_ellipse, z_ellipse, c = 'b', label = 'Ellipsoid')

        ax.legend()

        ax.set_aspect(1.0)

        ax = ax_arr[1]

        r_pts = np.sqrt((x**2.0 + y**2.0 + z**2.0))
        r_h = np.sqrt(x**2.0 + y**2.0)
        
        theta_pts = np.arctan2(r_h, z)
        cos_theta_pts = np.cos(theta_pts)
        
        r_from_theta_pts = r_cmb*(1.0 - ((2.0/3.0)*ellipticity_cmb*LegendrePoly2(cos_theta_pts)))

        max_r_error = np.max(np.abs(r_from_theta_pts - r_pts))
        print('Maximum r error = {:>.3e} (units of R_cmb)'.format(max_r_error/r_cmb))

        ax.scatter(r_pts - r_cmb, r_from_theta_pts - r_cmb, s = 3, c = 'k')
        ax.plot([-15.0, 0.0], [-15.0, 0.0])

        ax.set_xlabel('Radii of points - CMB radius')
        ax.set_ylabel('Expected radii at point latitude - CMB radius')

        ax.set_aspect(1.0)
        
        # -----------------------------------------------------------------

        # Plot points on 2D map.
        fig = plt.figure(figsize = (11.0, 8.5))
        ax = plt.gca()

        i_east = np.where(lon > 0.0)[0]
        n_tri = tri.shape[1]
        mask = np.zeros(n_tri, dtype = np.bool)
        mask[:] = True
        for i in range(n_tri):

            lon_deg_i = lon_deg[tri[:, i]]
            lon_diffs_i = [lon_deg_i[0] - lon_deg_i[1], lon_deg_i[0] - lon_deg_i[2],
                            lon_deg_i[1] - lon_deg_i[2]]
            max_lon_diff = np.max(np.abs(lon_diffs_i))

            if max_lon_diff < 180.0:

                mask[i] = False

        ax.triplot(lon_deg, lat_deg, triangles = tri.T, mask = mask)
        #ax.scatter(lon_deg[i_outline], lat_deg[i_outline], c = 'r', zorder = 10)
        ax.plot(lon_deg[i_outline], lat_deg[i_outline], c = 'r', marker = '.', linestyle = '-', zorder = 10)
        ax.scatter(lon_deg[i_inside], lat_deg[i_inside], c = 'g', zorder = 11)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xlim([-180.0, 180.0])
        ax.set_ylim([-90.0, 90.0])
        
        # Plot points in 3D.
        fig = plt.figure(figsize = (8.5, 8.5))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_trisurf(x, y, z, triangles = tri.T, edgecolor = (1.0, 0.0, 0.0, 0.2), color = (1.0, 1.0, 1.0, 1.0)) #cmap = 'magma') 
        ax.plot(x[i_outline], y[i_outline], z[i_outline], zorder = 10)
        ax.scatter(x[i_inside], y[i_inside], z[i_inside], color = 'g', zorder = 11)

        plt.show()

    return pts, tri, i_inside, i_outline

def move_points_outwards_ellipsoidal(pts, d_r_spherical, r_spherical, ellipticity_at_d_r):
    '''
    r_spherical     Known spherical radius of points (before flattening).
    ellipticity  Ellipticity at radius r + d_r.
    '''

    # Get the longitude and latitude of the points.
    _, lon, lat = XYZ_to_RLonLat(*pts)

    # Add the outward offset.
    r_new_spherical = r_spherical + d_r_spherical

    # Find the deformed coordinates.
    x, y, z = RLonLatEll_to_XYZ(r_new_spherical, lon, lat, ellipticity_at_d_r)

    pts = np.array([x, y, z])

    return pts

def move_points_outwards_spherical(pts, d_r):

    # Convert to geographical coordinates.
    r, lon, lat = XYZ_to_RLonLat(*pts)

    # Add the outward offset.
    r = r + d_r

    # Convert back to Cartesian coordinates.
    x, y, z = RLonLat_to_XYZ(r, lon, lat)
    pts = np.array([x, y, z])

    return pts

def rescale_to_ellipsoid(pts, r, ellipticity):
    
    x, y, z = pts
    r_pts = np.sqrt(x**2.0 + y**2.0 + z**2.0)

    r_h_pts = np.sqrt(x**2.0 + y**2.0)
    theta_pts = np.arctan2(r_h_pts, z)
    cos_theta_pts = np.cos(theta_pts)
    
    r_from_theta_pts = r*(1.0 - ((2.0/3.0)*ellipticity*LegendrePoly2(cos_theta_pts)))
    scale_factor = r_from_theta_pts/r_pts

    pts = pts*scale_factor

    return pts

def rescale_to_surface_old(pts, r, ellipticity):

    r_pts = np.linalg.norm(pts, axis = 0)
    if ellipticity is None:

        scale_factor = r/r_pts
    
    else:
        
        #r_h_pts = np.linalg.norm(pts[:, 0:2], axis = 1) 
        r_h_pts = np.sqrt(pts[0, :]**2.0 + pts[1, :]**2.0)
        theta_pts = np.arctan2(r_h_pts, pts[2, :])
        cos_theta_pts = np.cos(theta_pts)
    
        r_from_theta_pts = r*(1.0 - ((2.0/3.0)*ellipticity*LegendrePoly2(cos_theta_pts)))

        ##
        #r_diff = r_pts - r_from_theta_pts
        #max_r_diff = np.max(r_diff)
        #min_r_diff = np.min(r_diff)
        #print('Maximum positive deviation from spheroid: {:>+7.3f} km'.format(max_r_diff))
        #print('Maximum negative deviation from spheroid: {:>-7.3f} km'.format(min_r_diff))

        scale_factor = r_pts/r_from_theta_pts

    #
    pts = pts*scale_factor

    return pts

def create_spherical_shell(geom, mesh_size, r):

    spherical_shell = geom.add_ball([0.0, 0.0, 0.0], r, mesh_size = mesh_size)

    return spherical_shell

def create_ellipsoid_shell_old(geom, mesh_size, r, ellipticity):
    '''
    Creates a spheroid with a true elliptical boundary, not approximate.
    '''

    # Define the three semi-major axes of the ellipsoid along the
    # x-, y- and z- axes.
    if ellipticity is None:

        # Spherical case: Three axes are the same.
        r_x = r
        r_y = r_x
        r_z = r_x

    else:

        # Ellipsoidal case: Axes in the equatorial plane (x- and y- axes)
        # are greater than spherical radius, polar axis (z-axis) is
        # less.
        r_x = r*(1.0 + (ellipticity/3.0))
        r_y = r_x 
        r_z = r*(1.0 - ((2.0*ellipticity)/3.0))

    # Add the ellipsoidal surface.
    ellipsoid_shell = geom.add_ellipsoid([0.0, 0.0, 0.0], [r_x, r_y, r_z], mesh_size = mesh_size)

    return ellipsoid_shell

def make_mesh_sizing_function_spherical(subdir_out, tet_max_vol, tet_min_max_edge_length_ratio, r_icb, r_cmb, r_srf, name):

    # Determine the mesh size on the CMB sphere. 
    mesh_size_min = 0.5*(2.0**(1.0/2.0))*(3.0**(1.0/3.0))*(tet_max_vol**(1.0/3.0))
    #mesh_size_min = 3.0*mesh_size_min
    mesh_size_max = tet_min_max_edge_length_ratio*mesh_size_min

    # Set output file for mesh points. 
    #subdir_out = os.path.join(dir_output, 'sphere_{:>07.2f}'.format(mesh_size_min))
    #
    #name = 'spheres'
    path_node           = os.path.join(subdir_out, '{:}.b.node'.format(name))
    path_ele            = os.path.join(subdir_out, '{:}.b.ele'.format(name))
    path_edge_length    = os.path.join(subdir_out, '{:}.b.mtr'.format(name))

    # Load mesh points if already exist.
    path_list = [path_node, path_ele, path_edge_length]
    files_exist = all([os.path.exists(path_) for path_ in path_list])

    if files_exist:

        print('Mesh sizing files already exist, skipping.')
        return
    
    # Create a unit sphere for the concentric shells of the mesh sizing function.
    mesh_size_unit = 0.3
    pts_unit, tri_unit = build_sphere(None, 1.0, mesh_size_unit, None) 
    n_pts_unit = pts_unit.shape[1]

    # Calculate the gradient in mesh size with radial coordinate.
    r_gap_max = r_srf - r_cmb
    mesh_size_range = (mesh_size_max - mesh_size_min)
    mesh_size_gradient = mesh_size_range/r_gap_max

    # Define the radii of spheres in the mesh sizing function, increasing 
    # from the centre.
    r_list = np.array([ r_icb/2.0, r_icb, (r_icb + r_cmb)/2.0, r_cmb,
                        (r_cmb + r_srf)/2.0, r_srf, r_srf*1.05])
    mesh_size_list = np.array([
        mesh_size_gradient*(r_icb -   0.0)*0.5,   0.0,
        mesh_size_gradient*(r_cmb - r_icb)*0.5,   0.0,
        mesh_size_gradient*(r_srf - r_cmb)*0.5,   0.0,
        0.0])
    mesh_size_list = mesh_size_list + mesh_size_min

    # Start with a single point at the centre.
    pts = np.atleast_2d([0.0, 0.0, 0.0]).T
    mesh_size = np.atleast_1d(mesh_size_min)

    # Add points from the concentric spheres.
    n_sphere = len(r_list)
    for i in range(n_sphere):

        pts_sphere = pts_unit*r_list[i]
        mesh_size_sphere = (np.zeros(n_pts_unit) + 1.0)*mesh_size_list[i]

        pts = np.append(pts, pts_sphere, axis = 1)
        mesh_size = np.append(mesh_size, mesh_size_sphere)

    # Tetrahedralise the points.
    tri_obj = Delaunay(pts.T)
    tri = tri_obj.simplices.T
    tri = tri + 1 # Convert to 1-based indexing.

    # Get info.
    n_pts = pts.shape[1]
    n_tri = tri.shape[1]

    # Save the node file.
    print('Saving to {:}'.format(path_node))
    with open(path_node, 'w') as out_id:

        out_id.write('{:>12d} {:>12d} {:>12d} {:>12d}\n'.format(n_pts, 3, 0, 0))

        for i in range(n_pts):

            out_id.write('{:>12d} {:>12.5e} {:>12.5e} {:12.5e}\n'.format(i + 1, pts[0, i], pts[1, i], pts[2, i]))

    # Save the tetrahedral element file.
    print('Saving to {:}'.format(path_ele))
    with open(path_ele, 'w') as out_id:

        out_id.write('{:>12d} {:>12d} {:>12d}\n'.format(n_tri, 4, 0))

        for i in range(n_tri):

            out_id.write('{:>12d} {:>12d} {:>12d} {:>12d} {:>12d}\n'.format(i + 1, tri[0, i], tri[1, i], tri[2, i], tri[3, i]))

    # Save the edge length file.
    print('Saving to {:}'.format(path_edge_length))
    with open(path_edge_length, 'w') as out_id:

        out_id.write('{:>12d} {:>12d}\n'.format(n_pts, 1))

        for i in range(n_pts):

            out_id.write('{:>12.5e}\n'.format(mesh_size[i]))

    return

def make_mesh_sizing_function_ellipsoidal(subdir_out, tet_max_vol, tet_min_max_edge_length_ratio, model, discon_lists, name, ellipticity_data, mesh_size_maxima):

    r_ellipticity_profile = ellipticity_data[:, 0]
    r_ellipticity_profile = r_ellipticity_profile*1.0E-3 # Convert to km.
    ellipticity_profile = ellipticity_data[:, 1]

    # Determine the mesh size on the CMB sphere. 
    mesh_size_min = 0.5*(2.0**(1.0/2.0))*(3.0**(1.0/3.0))*(tet_max_vol**(1.0/3.0))
    #mesh_size_min = mesh_size_min/3.0
    #mesh_size_min = 3.0*mesh_size_min
    mesh_size_max = tet_min_max_edge_length_ratio*mesh_size_min

    #mesh_size_maxima = mesh_size_maxima/3.0

    # Set output file for mesh points. 
    #subdir_out = os.path.join(dir_output, 'sphere_{:>07.2f}'.format(mesh_size_min))
    #
    #name = 'spheres'
    path_node           = os.path.join(subdir_out, '{:}.b.node'.format(name))
    path_ele            = os.path.join(subdir_out, '{:}.b.ele'.format(name))
    path_edge_length    = os.path.join(subdir_out, '{:}.b.mtr'.format(name))

    # Load mesh points if already exist.
    path_list = [path_node, path_ele, path_edge_length]
    files_exist = all([os.path.exists(path_) for path_ in path_list])

    if files_exist:

        print('Mesh sizing files already exist, skipping.')
        return

    # Define the spherical radii of spheres in the mesh sizing function, increasing 
    # from the centre.
    r_discons = model['r'][discon_lists['all']]
    r_discons_with_zero = np.insert(r_discons, 0, 0.0)
    n_discons = len(r_discons)
    n_shells = 2*n_discons + 1
    r_shells = np.zeros(n_shells)
    for i in range(n_discons):

        r_shells[2*i] = r_discons_with_zero[i] + (r_discons_with_zero[i + 1] - r_discons_with_zero[i])/2.0
        r_shells[(2*i) + 1] = r_discons_with_zero[i + 1]

    r_shells[-1] = 1.05*r_shells[-2]

    # Calculate the gradient in mesh size with radial coordinate.
    r_gap_max = np.max(np.diff(r_shells))

    mesh_size_range = (mesh_size_max - mesh_size_min)
    mesh_size_gradient = mesh_size_range/r_gap_max

    mesh_size_list = np.zeros(n_shells)
    for i in range(n_discons):
        
        mesh_size_mid = (r_shells[(2*i) + 1] - r_shells[(2*i)])*mesh_size_gradient
        mesh_size_list[(2*i)] = mesh_size_mid

    mesh_size_list = mesh_size_list + mesh_size_min

    for i in range(n_discons):
        
        if mesh_size_list[2*i + 1] > mesh_size_maxima[i]:

            mesh_size_list[2*i + 1] = mesh_size_maxima[i]

        if mesh_size_list[2*i] > mesh_size_maxima[i]:

            mesh_size_list[2*i] = mesh_size_maxima[i]

    mesh_size_list[-1] = mesh_size_list[-2]

    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax =  plt.gca()

    #ax.plot(r_shells, mesh_size_list)

    #plt.show()

    #sys.exit()

    # Find the ellipticity of each of the shells.
    ellipticity_list = np.interp(r_shells, r_ellipticity_profile, ellipticity_profile)
    
    # Create a unit sphere for each of the concentric shells of the mesh sizing function.
    n_shells = len(r_shells)
    mesh_size_unit = 0.3
    pts_unit_list = []

    for i in range(n_shells):

        pts_unit, tri_unit = build_ellipsoid(None, 1.0, mesh_size_unit, ellipticity_list[i]) 
        #n_pts_unit = pts_unit.shape[1]

        pts_unit_list.append(pts_unit)

    # Start with a single point at the centre.
    pts = np.atleast_2d([0.0, 0.0, 0.0]).T
    mesh_size = np.atleast_1d(mesh_size_min)

    # Add points from the concentric spheres.
    for i in range(n_shells):

        pts_shell = pts_unit_list[i]*r_shells[i]
        n_pts_unit = pts_unit_list[i].shape[1]
        mesh_size_shell = (np.zeros(n_pts_unit) + 1.0)*mesh_size_list[i]

        pts = np.append(pts, pts_shell, axis = 1)
        mesh_size = np.append(mesh_size, mesh_size_shell)

    # Tetrahedralise the points.
    tri_obj = Delaunay(pts.T)
    tri = tri_obj.simplices.T
    tri = tri + 1 # Convert to 1-based indexing.

    # Get info.
    n_pts = pts.shape[1]
    n_tri = tri.shape[1]

    # Save the node file.
    print('Saving to {:}'.format(path_node))
    with open(path_node, 'w') as out_id:

        out_id.write('{:>12d} {:>12d} {:>12d} {:>12d}\n'.format(n_pts, 3, 0, 0))

        for i in range(n_pts):

            out_id.write('{:>12d} {:>12.5e} {:>12.5e} {:12.5e}\n'.format(i + 1, pts[0, i], pts[1, i], pts[2, i]))

    # Save the tetrahedral element file.
    print('Saving to {:}'.format(path_ele))
    with open(path_ele, 'w') as out_id:

        out_id.write('{:>12d} {:>12d} {:>12d}\n'.format(n_tri, 4, 0))

        for i in range(n_tri):

            out_id.write('{:>12d} {:>12d} {:>12d} {:>12d} {:>12d}\n'.format(i + 1, tri[0, i], tri[1, i], tri[2, i], tri[3, i]))

    # Save the edge length file.
    print('Saving to {:}'.format(path_edge_length))
    with open(path_edge_length, 'w') as out_id:

        out_id.write('{:>12d} {:>12d}\n'.format(n_pts, 1))

        for i in range(n_pts):

            out_id.write('{:>12.5e}\n'.format(mesh_size[i]))

    return

def make_ellipsoidal_poly_file(pts_list_spheres, pts_llsvp_top, tri_list_spheres, tri_llsvp_top, quad_llsvp_sides, pts_regions, path_poly): 
    '''
    See section 5.2.2 of the TetGen manual for the specification of the .poly
    file.
    '''

    # Merge point lists.
    pts = np.concatenate([*pts_list_spheres, pts_llsvp_top], axis = 1)

    # Get information about mesh.
    n_dim = pts.shape[0]
    n_nodes = pts.shape[1]
    #n_tri = tri.shape[1]
    n_tri_spheres_list = [tri_i.shape[1] for tri_i in tri_list_spheres]
    n_tri_spheres = sum(n_tri_spheres_list)
    n_tri_llsvp_top = tri_llsvp_top.shape[1]
    n_quad_llsvp_sides = quad_llsvp_sides.shape[1]
    n_facets = n_tri_spheres + n_tri_llsvp_top + n_quad_llsvp_sides
    #
    n_regions = pts_regions.shape[1]

    # Shift from 0-based to 1-based indexing.
    n_spheres = len(tri_list_spheres)
    for i in range(n_spheres):
        tri_list_spheres[i] = tri_list_spheres[i] + 1
    tri_llsvp_top = tri_llsvp_top + 1
    quad_llsvp_sides = quad_llsvp_sides + 1
    
    # Header lines for part one: Node list.
    comments_1a = ['# Part 1 - node list', '# node count, 3 dim, no attribute, no boundary marker']
    comment_1b = '# Node index, node coordinates'
    header_1 = '{:>8d}{:>8d}{:>8d}{:>8d}'.format(n_nodes, n_dim, 0, 0)
    header_and_comments_1 = [*comments_1a, header_1, comment_1b]

    # Header lines for part two: Facet list.
    comments_2a = ['# Part 2 - facet list', '# facet count, no boundary marker']
    header_2 = '{:>8d}{:>8d}'.format(n_facets, 0)
    comment_2b = '# facets'
    header_and_comments_2 = [*comments_2a, header_2, comment_2b]

    # Lines for parts 3 and 4 (hole and region lists).
    comment_3 = '# Part 3 - hole list'
    header_3 = '{:>8d}'.format(0)
    #
    comment_4 = '# Part 4 - region list'
    header_4 = '{:>8d}'.format(n_regions)
    #
    header_and_comments_3 = [comment_3, header_3]
    header_and_comments_4 = [comment_4, header_4]

    # Write the file.
    print('Writing .poly file {:}'.format(path_poly))
    with open(path_poly, 'w') as out_id:

        # Write header lines for part 1.
        for line in header_and_comments_1: 

            out_id.write('{:}\n'.format(line))
        
        i_pts = 0
        for j in range(n_spheres):

            n_nodes_j = pts_list_spheres[j].shape[1]
            out_id.write('# Nodes of shell {:>d}\n'.format(j))

            # Write the node list for this shell. 
            for i in range(n_nodes_j):

                # Note: Write (i + 1) instead of (i) because of 1-based indexing.
                out_id.write('{:>8d} {:>+20.14e} {:>+20.14e} {:>+20.14e}\n'.format(i_pts + 1, pts_list_spheres[j][0, i], pts_list_spheres[j][1, i], pts_list_spheres[j][2, i]))

                i_pts = i_pts + 1

        n_nodes_llsvp_top = pts_llsvp_top.shape[1]
        out_id.write('# Nodes of top of LLSVP.\n'.format(j))

        # Write the node list for this shell. 
        for i in range(n_nodes_llsvp_top):

            # Note: Write (i + 1) instead of (i) because of 1-based indexing.
            out_id.write('{:>8d} {:>+20.14e} {:>+20.14e} {:>+20.14e}\n'.format(i_pts + 1, pts_llsvp_top[0, i], pts_llsvp_top[1, i], pts_llsvp_top[2, i]))

            i_pts = i_pts + 1

        # Blank line before part 2.
        out_id.write('\n')

        # Write header lines for part 2.
        for line in header_and_comments_2:

            out_id.write('{:}\n'.format(line))

        # Write the facet list for part 2.
        for j in range(n_spheres):

            out_id.write('# Facets (triangles) of shell {:>d}.\n'.format(j))
            n_tri_j = tri_list_spheres[j].shape[1]

            for i in range(n_tri_j):
                
                # Write '1' to indicate each facet is a single triangle.
                out_id.write('{:>8d}\n'.format(1))

                # Write 3, i0, i1, i2 where 3 is the number of vertices of the triangle,
                # and i0, i1, i2 are the indices of the vertices of the triangle.
                out_id.write('{:>8d} {:>8d} {:>8d} {:>8d}\n'.format(3, tri_list_spheres[j][0, i], tri_list_spheres[j][1, i], tri_list_spheres[j][2, i]))

        # Second: Facets of the upper surface of the LLSVP. 
        # Comment line.
        out_id.write('# Facets (triangles) of upper surface of LLSVP.\n')
        #
        for i in range(n_tri_llsvp_top):

            # Write '1' to indicate each facet is a single triangle.
            out_id.write('{:>8d}\n'.format(1))

            # Write 3, i0, i1, i2 where 3 is the number of vertices of the triangle,
            # and i0, i1, i2 are the indices of the vertices of the triangle.
            out_id.write('{:>8d} {:>8d} {:>8d} {:>8d}\n'.format(3, tri_llsvp_top[0, i], tri_llsvp_top[1, i], tri_llsvp_top[2, i]))

        # Third: Sides of the LLSVP.
        # Comment line.
        out_id.write('# Facets (quadrilaterals) of vertical sides of the LLSVP.\n')
        #
        for i in range(n_quad_llsvp_sides):

            # Write '1' to indicate each facet is a single quadrilateral. 
            out_id.write('{:>8d}\n'.format(1))

            # Write 4, i0, i1, i2, i3 where 4 is the number of vertices of the quadrilateral,
            # and i0, i1, i2, i3 are the indices of the vertices of the quadrilateral.
            out_id.write('{:>8d} {:>8d} {:>8d} {:>8d} {:>8d}\n'.format(4, quad_llsvp_sides[0, i], quad_llsvp_sides[1, i], quad_llsvp_sides[2, i], quad_llsvp_sides[3, i]))

        # Blank line before part 3.
        out_id.write('\n')

        # Write part 3.
        for line in header_and_comments_3:

            out_id.write('{:}\n'.format(line))

        # Blank line before part 4.
        out_id.write('\n')

        # Write part 4.
        for line in header_and_comments_4:

            out_id.write('{:}\n'.format(line))

        for i in range(n_regions):

            out_id.write('{:>8d} {:>+20.14e} {:>+20.14e} {:>+20.14e} {:>8d}\n'.format(i + 1,
                pts_regions[0, i], pts_regions[1, i], pts_regions[2, i], i + 1))

    return

def tetrahedralise_poly_file(tet_max_vol, path_poly, subdir_out, name):
    '''
    Generate the mesh by calling tetgen from command line.
    This creates .ele, .neigh and .node files.
        unix([tetgen,' -pq1.5nYVFAa', num2str(tet_max_vol,'%f'),' ', mod_path,'.poly']);
     p     Take a piecewise linear complex (.poly file) and tetrahedralise it.
     q1.5/20.0 Improve quality of mesh so that each element has no radius/edge
               ratio greater than 1.5, and dihedral angles no smaller than
               20.0
     n         Output tetrahedra neighbours to .neigh file.
     Y         Do not modify input surface meshes.
     V         Verbose.
     F         Do not output .face or .edge files.
     A         Assign attributes to tetrahedra in different regions.
     a         Apply a maximum tetrahedron volume constraint.
     k          Save a VTK file for visualisation.
     more
    %unix([tetgen,' -pq1.5/10.0nYVFAak', num2str(tet_max_vol,'%f'),' ', mod_path_llsvp,'.poly']);
    %unix([tetgen,' -pmq1.5/10.0nYVFAa', num2str(tet_max_vol,'%f'),' ', mod_path,'.poly']);
    unix([tetgen,' -A -pmq1.5/10.0nYCVFO5/7 -a ', num2str(tet_max_vol,'%f'),' ', mod_path_llsvp,'.poly']);
    %unix([tetgen,' -pYq1.5AmO5/7nFCV ', mod_path_llsvp,'.poly']);
    '''
    
    path_ele    = os.path.join(subdir_out, '{:}.1.ele'.format(name))
    path_neigh  = os.path.join(subdir_out, '{:}.1.neigh'.format(name))
    path_node   = os.path.join(subdir_out, '{:}.1.node'.format(name))

    paths = [path_ele, path_neigh, path_node]
    paths_exist = all([os.path.exists(path) for path in paths])

    if paths_exist:

        print('All TetGen files exist. Skipping mesh creation.')
        return

    radius_edge_ratio = 1.5 # 1.5 Smaller is better (more equant).
    max_dihedral_angle = 15.0 # 20.0 Smallest dihedral angle (larger is better).

    ##command = 'tetgen -A -pmq{:.2f}/{:.2f}nYCVFO5/7 -a {:>7.2f} {:}'.format(radius_edge_ratio, max_dihedral_angle, tet_max_vol, path_poly)
    #command = 'tetgen -k -A -pmq{:.2f}/{:.2f}nYCVFO5/7 -a {:>7.2f} {:}'.format(radius_edge_ratio, max_dihedral_angle, tet_max_vol, path_poly)
    command = 'tetgen -k -A -pmq{:.2f}/{:.2f}nCVFO5/7 -a {:>7.2f} {:}'.format(radius_edge_ratio, max_dihedral_angle, tet_max_vol, path_poly)
    ##command = 'tetgen -k -A -pmnYCVFO5/7 -a {:>7.2f} {:}'.format(tet_max_vol, path_poly)
    print(command)
    os.system(command)

    return 

# Assigning parameters at mesh points. ----------------------------------------
def assign_parameters(dir_input, subdir_out, name, order, model, discon_lists, ellipticity_data):
    
    files_exist = check_files_assign_params(subdir_out, name, order)
    if files_exist:

        print('All model parameter files exist, skipping.')
        return

    # Define file paths.
    path_node   = os.path.join(subdir_out, '{:}.1.node'.format(name))
    path_ele    = os.path.join(subdir_out, '{:}.1.ele'.format(name))
    path_neigh  = os.path.join(subdir_out, '{:}.1.neigh'.format(name))

    # Read the nodes and elements (currently no need to load neighbours).
    nodes, tets, tet_labels, neighs = load_tetgen_mesh(path_node, path_ele, path_neigh) 
    
    # Get the tetrahedron information (number of points, geometric factors
    # defining coordinates of points).
    tet_info = get_tet_info(order)

    # Get the coordinates of each point in the tetrahedral mesh (including
    # higher-order points if requested).
    nodal_pts, links = get_nodal_points(nodes, tets, tet_info)
    
    #links2 = links.copy()
    ##links = links + 1
    #n_tets = tets.shape[1]
    #links = np.zeros((tet_info['n_pts'], n_tets), dtype = np.int)
    #k = 0 
    #for i in range(n_tets):
    #    for j in range(tet_info['n_pts']):
    #        links[j, i] = k
    #        k = k + 1

    #print(links[:, 0:10])
    #print(links2[:, 0:10])

    ##assert np.all(links == links2)
    ##print(tets.shape)
    ##print(order)
    #print(links.shape)
    #print(links2.shape)

    #sys.exit()

    ##print(links[:, -5:])

    # Create the output arrays.
    n_tets = tets.shape[1]
    n_nodes_per_tet = nodal_pts.shape[1]
    #
    v_p = np.zeros((n_nodes_per_tet, n_tets))
    v_s = np.zeros((n_nodes_per_tet, n_tets))
    rho = np.zeros((n_nodes_per_tet, n_tets))
    
    # Make a list of discontinuity indices with 0 included.
    i_discon_with_zero = np.insert(discon_lists['all'], 0, -1)
    
    # Write a summary of the number of nodes and tetrahedra in each region.
    file_summary = 'mesh_info.txt' 
    path_summary = os.path.join(subdir_out, file_summary)
    print('Writing mesh summary: {:}'.format(path_summary))
    with open(path_summary, 'w') as out_id:
       
        out_id.write('{:>6} {:>10} {:>10}\n'.format('Region', 'Nodes', 'Tetrahedra'))
        for j in range(discon_lists['n_discons'] + 1):

            # Find the tetrahedron in the specified region.
            #label = label_list[j]
            label = j + 1
            k = np.where(tet_labels == label)[0]
            n_tets_k = len(k)
            n_pts_k = len(np.unique(tets[:, k].flatten()))

            out_id.write('{:>6d} {:>10d} {:>10d}\n'.format(label, n_pts_k, n_tets_k))

        n_pts = nodes.shape[1]
        out_id.write('{:>6} {:>10d} {:>10d}\n'.format('All', n_pts, n_tets))

    # Interpolate the radial model at the nodal points.
    # Loop over the discontinuities and the LLSVP region (n_discons + 1).
    label_llsvp = discon_lists['n_discons'] + 1
    for j in range(discon_lists['n_discons'] + 1):

        # Find the tetrahedron in the specified region.
        #label = label_list[j]
        label = j + 1
        k = np.where(tet_labels == label)[0]
        nodal_pts_region = nodal_pts[:, :, k]

        if ellipticity_data is None:

            r_pts_region = np.linalg.norm(nodal_pts_region, axis = 0)

        else:
            
            r_ellipticity_profile = ellipticity_data[:, 0]*1.0E-3
            ellipticity_profile = ellipticity_data[:, 1]
            r_pts_region, _ = XYZ_to_REll(*nodal_pts_region, r_ellipticity_profile, ellipticity_profile)

        # Extract the relevant portion of the reference model.
        if label == label_llsvp:

            i0 = i_discon_with_zero[discon_lists['j_cmb'] + 1] + 1
            i1 = i_discon_with_zero[discon_lists['j_cmb'] + 2] + 1

        else:

            i0 = i_discon_with_zero[j] + 1
            i1 = i_discon_with_zero[j + 1] + 1
        #
        r_mod_region = model['r'][i0 : i1]
        v_s_mod_region = model['v_s'][i0 : i1]
        v_p_mod_region = model['v_p'][i0 : i1]
        rho_mod_region = model['rho'][i0 : i1]

        # Interpolate at the coordinates of the region using the reference
        # model.
        v_p_pts_region = np.interp(r_pts_region, r_mod_region, v_p_mod_region)
        v_s_pts_region = np.interp(r_pts_region, r_mod_region, v_s_mod_region)
        rho_pts_region = np.interp(r_pts_region, r_mod_region, rho_mod_region)

        # Store the anomalies for this region in the global array.
        v_p[:, k] = v_p_pts_region
        v_s[:, k] = v_s_pts_region
        rho[:, k] = rho_pts_region

    # Apply the LLSVP anomalies.
    # First, create copies of the arrays with no anomalies.
    v_p_no_anomaly = v_p.copy()
    v_s_no_anomaly = v_s.copy()
    rho_no_anomaly = rho.copy()
    #
    v_p_anomaly = -0.016
    v_s_anomaly = -0.040
    rho_anomaly = +0.010
    #
    k = np.where(tet_labels == label_llsvp)[0]
    v_p[:, k] = v_p[:, k]*(1.0 + v_p_anomaly)
    v_s[:, k] = v_s[:, k]*(1.0 + v_s_anomaly)
    rho[:, k] = rho[:, k]*(1.0 + rho_anomaly)

    # Loop over each variable and save as a binary file.
    var_str_list = ['vp', 'vs', 'rho']
    arr_list_without_anomaly = [v_p_no_anomaly, v_s_no_anomaly, rho_no_anomaly]
    arr_list_with_anomaly    = [v_p, v_s, rho]
    arr_list = [arr_list_with_anomaly, arr_list_without_anomaly]
    anomaly_str_list = ['with_anomaly', 'without_anomaly']
    for i in range(3):

        for j in range(2):

            # Get file path.
            file_ = '{:}.1_{:}_pod_{:1d}_true_{:}.dat'.format(name, var_str_list[i], order, anomaly_str_list[j]) 
            path = os.path.join(subdir_out, file_)

            # Save as a binary file.
            print('Saving to {:}.'.format(path))
            with open(path, 'w') as out_id:

                arr_list[j][i].T.tofile(path)

    # Save the other input files in the binary format required by NormalModes.
    # First, save mesh.header file, which simply lists the number of
    # tetrahedra and the number of nodes.
    n_pts = nodes.shape[1]
    file_mesh_header = '{:}.1_mesh.header'.format(name)
    path_mesh_header = os.path.join(subdir_out, file_mesh_header) 
    print('Saving to {:}'.format(path_mesh_header))
    with open(path_mesh_header, 'w') as out_id:

        out_id.write('{:d} {:d}'.format(n_tets, n_pts))

    # Second, save the ele.dat file, which stores the tetrahedron indices
    # in binary format.
    file_ele_dat = '{:}.1_ele.dat'.format(name)
    path_ele_dat = os.path.join(subdir_out, file_ele_dat)
    print('Saving to {:}'.format(path_ele_dat))
    tets_int32 = tets.astype(np.int32) # Convert to 32-bit integer.
    tets_int32 = tets_int32 + 1 # Convert to 1-based indexing.
    with open(path_ele_dat, 'w') as out_id:

        # Note: Transpose required for correct flattening.
        (tets_int32.T).tofile(out_id)

    # Second, save the node.dat file, which stores the node coordinates.
    # in binary format.
    file_node_dat = '{:}.1_node.dat'.format(name)
    path_node_dat = os.path.join(subdir_out, file_node_dat)
    print('Saving to {:}'.format(path_node_dat))
    with open(path_node_dat, 'w') as out_id:

        # Note: Transpose required for correct flattening.
        (nodes.T).tofile(out_id)

    # Second, save the node.dat file, which stores the node coordinates.
    # in binary format.
    file_neigh_dat = '{:}.1_neigh.dat'.format(name)
    path_neigh_dat = os.path.join(subdir_out, file_neigh_dat)
    print('Saving to {:}'.format(path_neigh_dat))
    neighs_int32 = neighs.astype(np.int32) # Convert to 32-bit integer.
    neighs_int32 = neighs_int32 + 1 # Convert to 1-based indexing.
    with open(path_neigh_dat, 'w') as out_id:

        # Note: Transpose required for correct flattening.
        (neighs_int32.T).tofile(out_id)

    # Finally, save as a VTK file (only for visualisation).
    # Note: No file suffix is added because PyEVTK automatically adds the
    # suffix .vtu.
    if order == 1:

        path_vtk = os.path.join(subdir_out, '{:}_pOrd_{:>1d}'.format(name, order))
        save_model_to_vtk(path_vtk, nodes, tets, links, tet_labels, v_p, v_s, rho, order)

    else:

        print('Not saving model to VTK file, not implemented yet for order = {:>1d}.'.format(order))

    # Create symbolic links.
    # This allows the two models (with/without anomaly) to share their
    # node files.
    create_symlinks(subdir_out, name, order, file_node_dat, file_ele_dat, file_neigh_dat, file_mesh_header)

    return

def check_files_assign_params(subdir_out, name, order):

    path_list = []

    file_summary = 'mesh_info.txt' 
    path_summary = os.path.join(subdir_out, file_summary)
    path_list.append(path_summary)

    # Loop over each variable and save as a binary file.
    var_str_list = ['vp', 'vs', 'rho']
    anomaly_str_list = ['with_anomaly', 'without_anomaly']
    for i in range(3):

        for j in range(2):

            # Get file path.
            file_ = '{:}.1_{:}_pod_{:1d}_true_{:}.dat'.format(name, var_str_list[i], order, anomaly_str_list[j]) 
            path = os.path.join(subdir_out, file_)
            path_list.append(path)

    file_mesh_header = '{:}.1_mesh.header'.format(name)
    path_mesh_header = os.path.join(subdir_out, file_mesh_header) 

    file_ele_dat = '{:}.1_ele.dat'.format(name)
    path_ele_dat = os.path.join(subdir_out, file_ele_dat)

    file_node_dat = '{:}.1_node.dat'.format(name)
    path_node_dat = os.path.join(subdir_out, file_node_dat)

    file_neigh_dat = '{:}.1_neigh.dat'.format(name)
    path_neigh_dat = os.path.join(subdir_out, file_neigh_dat)

    path_list = path_list + [path_mesh_header, path_ele_dat, path_node_dat, path_neigh_dat]

    # Finally, save as a VTK file (only for visualisation).
    # Note: No file suffix is added because PyEVTK automatically adds the
    # suffix .vtu.
    if order == 1:

        path_vtk = os.path.join(subdir_out, '{:}_pOrd_{:>1d}.vtu'.format(name, order))
        path_list.append(path_vtk)

    # Create symbolic links.
    # This allows the two models (with/without anomaly) to share their
    # node files.
    path_list_symlink = create_symlinks(subdir_out, name, order, file_node_dat, file_ele_dat, file_neigh_dat, file_mesh_header, path_list_only = True)

    path_list = path_list + path_list_symlink
    files_exist = all([os.path.exists(path_) for path_ in path_list])

    return files_exist

def load_tetgen_mesh(path_node, path_ele, path_neigh):

    # Load the nodes.
    print('Loading mesh nodes from {:}'.format(path_node))
    nodes = np.loadtxt(path_node, skiprows = 1, comments = '#', usecols = (1, 2, 3)).T

    # Load the indices defining the tetrahedral elements, and the attribute of each element.
    print('Loading mesh elements from {:}'.format(path_ele))
    ele_attr = np.loadtxt(path_ele, skiprows = 1, comments = '#', usecols = (1, 2, 3, 4, 5), dtype = np.int).T
    tets = ele_attr[0:4, :]
    labels = ele_attr[4, :]

    # Convert to 0-based indexing.
    tets = tets - 1

    # Read the .neigh file, which lists the four neighbours (which share faces) of each tetrahedron. (Boundary faces have an index of -1.)
    # (See section 5.2.10 of the TetGen manual.)
    # [n_tet]   The number of tetrahedra.
    # neighs    (n_tet, 4) The indices of the neighbours of each tetrahedron.
    neighs      = np.loadtxt(
                    path_neigh,
                    comments    = '#',
                    skiprows    = 1,
                    usecols     = (1, 2, 3, 4),
                    dtype       = np.int).T
    # Note: Here switch to 0-based indexing (note: shifting indices by 1
    # means that boundary faces now have an index of -2).
    neighs = neighs - 1

    return nodes, tets, labels, neighs 

def get_tet_info(order):
    
    tet_info = dict()

    if order == 1:

        tet_info['n_pts'] = 4

        tet_info['r'] = np.array([-1.0, +1.0, -1.0, -1.0])
        tet_info['s'] = np.array([-1.0, -1.0, +1.0, -1.0])
        tet_info['t'] = np.array([-1.0, -1.0, -1.0, +1.0])

        tet_info['i_subelements'] = np.array([0, 1, 2, 3], dtype = np.int)
        tet_info['n_subelements'] = 1 

    elif order == 2:

        tet_info['n_pts'] = 10 

        tet_info['r'] = np.array([-1.0,  0.0, +1.0, -1.0,  0.0, -1.0, -1.0,  0.0, -1.0, -1.0])
        tet_info['s'] = np.array([-1.0, -1.0, -1.0,  0.0,  0.0, +1.0, -1.0, -1.0,  0.0, -1.0])
        tet_info['t'] = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  0.0,  0.0,  0.0, +1.0])

        tet_info['i_subelements'] = np.array([  [2,     3,     5,      8],
                                                [4,     5,     6,      9],
                                                [7,     8,     9,     10],
                                                [1,     2,     4,      7],
                                                [7,     2,     4,      8],
                                                [4,     2,     5,      8],
                                                [4,     8,     5,      9],
                                                [7,     8,     4,      9]],
                                                dtype = np.int)
        tet_info['i_subelements'] = tet_info['i_subelements'] - 1
        tet_info['n_subelements'] = 8 

    else:

        raise NotImplementedError

    return tet_info

def get_nodal_points(nodes, tets, tet_info):

    # Define geometric factors.
    r = tet_info['r']
    s = tet_info['s']
    t = tet_info['t']
    # Note: Shape of these terms must be (n_pts_per_tet, 1) for successful
    # broadcasting, so we use at_least2d and a transpose.
    a = np.atleast_2d(-1.0*(1.0 + r + s + t)).T
    b = np.atleast_2d((1.0 + r))             .T
    c = np.atleast_2d((1.0 + s))             .T
    d = np.atleast_2d((1.0 + t))             .T

    # Calculate coordinates of nodal points from corners of tetrahedra and
    # the geometric factors.
    x = 0.5*( a*nodes[0, tets[0, :]] + b*nodes[0, tets[1, :]]
            + c*nodes[0, tets[2, :]] + d*nodes[0, tets[3, :]])

    y = 0.5*( a*nodes[1, tets[0, :]] + b*nodes[1, tets[1, :]]
            + c*nodes[1, tets[2, :]] + d*nodes[1, tets[3, :]])

    z = 0.5*( a*nodes[2, tets[0, :]] + b*nodes[2, tets[1, :]]
            + c*nodes[2, tets[2, :]] + d*nodes[2, tets[3, :]])

    # Get a list of indices of sub-elements.
    n_tets = tets.shape[1]
    n_sub = tet_info['n_subelements']
    n_links = n_tets*n_sub
    offset = 0
    links = np.zeros((n_links, 4), dtype = np.int)
    for i in range(n_tets):

        i0 = i*n_sub
        i1 = (i + 1)*n_sub

        links[i0 : i1, :] = tet_info['i_subelements'] + offset
        offset = offset + tet_info['n_pts'] 

    links = links.T
    
    nodal_pts = np.array([x, y, z])

    return nodal_pts, links

def load_radial_model(path_model):
    
    # Load the model.
    model = dict()
    model['r'], model['rho'], model['v_p'], model['v_s'] = np.loadtxt(path_model).T

    # Find the indices of the discontinuities.
    i_discon = dict()
    r_diffs = np.abs(np.diff(model['r']))
    tolerance = 1.0E-10
    condition_discon_r = (r_diffs < tolerance)
    keys = ['rho', 'v_p', 'v_s']
    for key in keys:
        
        abs_diff = np.abs(np.diff(model[key]))
        condition_discon_key = (abs_diff > tolerance)
        
        i_discon[key] = np.where((condition_discon_r) & (condition_discon_key))[0] 

    # Merge the discontinuity lists.
    i_discon_merge = []
    for key in keys:

        i_discon_merge = i_discon_merge + list(i_discon[key])

    i_discon_merge = np.sort(np.unique(i_discon_merge)) 
    
    # Also include the outer point.
    n_pts = len(model['r'])
    i_discon_merge = np.append(i_discon_merge, n_pts - 1)

    # Find the indices of the fluid outer core.
    i_fluid = np.where(model['v_s'] < tolerance)[0]
    i_cmb = i_fluid[-1] + 1 # Index of lowermost layer in mantle.
    i_icb = i_fluid[0] # Index of lowermost layer in outer core.
    ##
    #n_layers = len(r)
    #i_inner_core = np.array(list(range(0, i_icb)), dtype = np.int)
    #i_outer_core = np.array(list(range(i_icb, i_cmb)), dtype = np.int)
    #i_mantle = np.array(list(range(i_cmb, n_layers)), dtype = np.int)

    # Find which discontinuity in the list is the CMB.
    j_discon_cmb = np.where((i_discon_merge == (i_cmb - 1)))[0][0]
    i_discon_cmb = (i_cmb - 1)
    i_discon_core = i_discon_merge[0 : j_discon_cmb]
    i_discon_mantle = i_discon_merge[j_discon_cmb + 1 : ]

    # Create dictionary.
    discon_lists = dict()
    discon_lists['all'] = i_discon_merge
    discon_lists['core'] = i_discon_core
    discon_lists['mantle'] = i_discon_mantle
    discon_lists['cmb'] = i_discon_merge[j_discon_cmb]
    discon_lists['j_cmb'] = j_discon_cmb
    n_discons = len(i_discon_merge)
    discon_lists['n_discons'] = n_discons

    return model, discon_lists

def create_symlinks(subdir_out, name, order, file_node_dat, file_ele_dat, file_neigh_dat, file_mesh_header, path_list_only = False):

    #path_anomaly_symmesh_header        = os.path.join('..', file_mesh_header)
    #path_anomaly_symlink_node_dat   = os.path.join('..', file_node_dat)
    #path_anomaly_symlink_ele_dat    = os.path.join('..', file_ele_dat)
    #path_anomaly_symlink_neigh_dat  = os.path.join('..', file_neigh_dat)

    # Define the symlink names for the non-common files.
    file_symlink_rho_dat = '{:}.1_rho_pod_{:1d}_true.dat'.format(name, order)
    file_symlink_v_p_dat = '{:}.1_vp_pod_{:1d}_true.dat'.format(name, order)
    file_symlink_v_s_dat = '{:}.1_vs_pod_{:1d}_true.dat'.format(name, order)

    # Define the common symlink targets.
    rel_path_node_dat       = os.path.join('../..', file_node_dat)
    rel_path_ele_dat        = os.path.join('../..', file_ele_dat)
    rel_path_neigh_dat      = os.path.join('../..', file_neigh_dat)
    rel_path_mesh_header    = os.path.join('../..', file_mesh_header)

    # Define the symlink targets for the model with anomaly.
    file_anomaly_rho_dat = '{:}.1_rho_pod_{:1d}_true_with_anomaly.dat'.format(name, order)
    file_anomaly_v_p_dat = '{:}.1_vp_pod_{:1d}_true_with_anomaly.dat'.format(name, order)
    file_anomaly_v_s_dat = '{:}.1_vs_pod_{:1d}_true_with_anomaly.dat'.format(name, order)
    #
    rel_path_anomaly_rho_dat = os.path.join('../..', file_anomaly_rho_dat)
    rel_path_anomaly_v_p_dat = os.path.join('../..', file_anomaly_v_p_dat)
    rel_path_anomaly_v_s_dat = os.path.join('../..', file_anomaly_v_s_dat)

    # Define the symlink targets for the model without anomaly.
    file_no_anomaly_rho_dat = '{:}.1_rho_pod_{:1d}_true_without_anomaly.dat'.format(name, order)
    file_no_anomaly_v_p_dat = '{:}.1_vp_pod_{:1d}_true_without_anomaly.dat'.format(name, order)
    file_no_anomaly_v_s_dat = '{:}.1_vs_pod_{:1d}_true_without_anomaly.dat'.format(name, order)
    #
    rel_path_no_anomaly_rho_dat = os.path.join('../..', file_no_anomaly_rho_dat)
    rel_path_no_anomaly_v_p_dat = os.path.join('../..', file_no_anomaly_v_p_dat)
    rel_path_no_anomaly_v_s_dat = os.path.join('../..', file_no_anomaly_v_s_dat)

    dir_pOrder = os.path.join(subdir_out, 'pOrder_{:>1d}'.format(order))
    mkdir_if_not_exist(dir_pOrder)
    dir_with_anomaly = os.path.join(dir_pOrder, 'with_anomaly')
    dir_without_anomaly = os.path.join(dir_pOrder, 'without_anomaly')
    if path_list_only:
        
        files = [   file_symlink_rho_dat, file_symlink_v_p_dat, file_symlink_v_s_dat,
                    file_node_dat, file_ele_dat, file_neigh_dat, file_mesh_header] 

        path_list_with_anomaly = [os.path.join(dir_with_anomaly, file_) for file_ in files]
        path_list_without_anomaly = [os.path.join(dir_without_anomaly, file_) for file_ in files]

        path_list = path_list_with_anomaly + path_list_without_anomaly

        return path_list

    # Pair the symlink targets with the symlink names for the model with
    # anomaly.
    path_pairs_with_anomaly =\
                [   [rel_path_node_dat,             file_node_dat], 
                    [rel_path_ele_dat,              file_ele_dat], 
                    [rel_path_neigh_dat,            file_neigh_dat],
                    [rel_path_mesh_header,          file_mesh_header],
                    [rel_path_anomaly_rho_dat,      file_symlink_rho_dat],
                    [rel_path_anomaly_v_p_dat,      file_symlink_v_p_dat], 
                    [rel_path_anomaly_v_s_dat,      file_symlink_v_s_dat]]

    # Pair the symlink targets with the symlink names for the model without
    # anomaly.
    path_pairs_without_anomaly =\
                [   [rel_path_node_dat,                 file_node_dat], 
                    [rel_path_ele_dat,                  file_ele_dat], 
                    [rel_path_neigh_dat,                file_neigh_dat],
                    [rel_path_mesh_header,              file_mesh_header],
                    [rel_path_no_anomaly_rho_dat,      file_symlink_rho_dat],
                    [rel_path_no_anomaly_v_p_dat,      file_symlink_v_p_dat], 
                    [rel_path_no_anomaly_v_s_dat,      file_symlink_v_s_dat]]

    # Create the symlinks.
    create_symlinks_cmd(path_pairs_with_anomaly, dir_with_anomaly)
    create_symlinks_cmd(path_pairs_without_anomaly, dir_without_anomaly)

    return

def create_symlinks_gravity(subdir_out, name, order, path_list_only = False):

    file_symlink_gravity_dat = '{:}.1_pod_{:1d}_potential_acceleration_true.dat'.format(name, order)

    file_no_anomaly_gravity_dat = '{:}.1_pod_{:1d}_potential_acceleration_true_without_anomaly.dat'.format(name, order)
    rel_path_no_anomaly_gravity_dat = os.path.join('../..', file_no_anomaly_gravity_dat)

    file_anomaly_gravity_dat = '{:}.1_pod_{:1d}_potential_acceleration_true_with_anomaly.dat'.format(name, order)
    rel_path_anomaly_gravity_dat = os.path.join('../..', file_anomaly_gravity_dat)

    dir_pOrder = os.path.join(subdir_out, 'pOrder_{:>1d}'.format(order))
    mkdir_if_not_exist(dir_pOrder)
    dir_with_anomaly = os.path.join(dir_pOrder, 'with_anomaly')
    dir_without_anomaly = os.path.join(dir_pOrder, 'without_anomaly')
    if path_list_only:

        path_list = [
                os.path.join(dir_with_anomaly, file_symlink_gravity_dat),
                os.path.join(dir_without_anomaly, file_symlink_gravity_dat)]
        
        return path_list


    path_pairs_with_anomaly = [[rel_path_anomaly_gravity_dat, file_symlink_gravity_dat]]
    path_pairs_without_anomaly = [[rel_path_no_anomaly_gravity_dat, file_symlink_gravity_dat]]

    # Create the symlinks.
    create_symlinks_cmd(path_pairs_with_anomaly, dir_with_anomaly)
    create_symlinks_cmd(path_pairs_without_anomaly, dir_without_anomaly)

    return

def create_symlinks_cmd(path_pairs, directory):

    # Record starting directory.
    start_dir = os.getcwd()

    # Change to symlink directory.
    mkdir_if_not_exist(directory)
    os.chdir(directory)

    # Create the symlinks.
    for path_pair in path_pairs:

        path_true = path_pair[0]
        path_symlink = path_pair[1]

        command = 'ln -sf {:} {:}'.format(path_true, path_symlink)
        print(command)
        os.system(command)

    # Go back to starting directory.
    os.chdir(start_dir)

    return

def save_model_to_vtk(path_vtk, pts, tets, links, tet_labels, v_p, v_s, rho, order):
    
    # Get number of tetrahedra.
    n_tets = tets.shape[1]

    # Define offsets (in the flattened tetrahedra connectivity list, the
    # index of the last node of each tetrahedron). Node 1-based indexing
    # is used for this list.
    if order == 1:

        n_pts_per_tet = 4
        cell_id = VtkTetra.tid

    elif order == 2:

        n_pts_per_tet = 10
        cell_id = VtkQuadraticTetra.tid

    else:

        raise ValueError
    
    offsets = n_pts_per_tet*(np.array(list(range(n_tets)), dtype = np.int) + 1)
    
    # VTK does not support discontinuities (where a shared node can have different
    # parameter values depending on which tetrahedron it is in). Therefore,
    # we must create a list of all the vertices of all tetrahedra (including
    # many repeating points).
    pts = pts[:, tets]

    # Unpack the points and make C-contiguous copies (necessary for the
    # PyEVTK package to write the .vtk file).
    x, y, z = pts
    #x = x.copy(order = 'C')
    #y = y.copy(order = 'C')
    #z = z.copy(order = 'C')

    # Flatten the lists in the correct order so that the indices for a
    # given tetrahedron appear sequentially.
    x       = x     .flatten(order = 'F')
    y       = y     .flatten(order = 'F')
    z       = z     .flatten(order = 'F')
    tets    = tets  .flatten(order = 'F')
    links   = links .flatten(order = 'F')
    v_p     = v_p   .flatten(order = 'F')
    v_s     = v_s   .flatten(order = 'F')
    rho     = rho   .flatten(order = 'F')
    #
    #v_s     = v_s.copy(order = 'C')

    # Define a list of cell types.
    # In this case, all of the cell types are the same (VtkTetra).
    cell_types = np.zeros(n_tets, dtype = np.int) + cell_id

    # Create a dictionary with information about the points.
    point_data = {'v_p' : v_p, 'v_s' : v_s, 'rho' : rho}

    # Create a dictionary with information about the tetrahedra.
    # Each array must be C-contiguous for the PyEVTK package.
    tet_labels = tet_labels.copy(order = 'C')
    tet_data = {"region" : tet_labels} 

    # Save the VTK file.
    print('Saving to {:}.vtu'.format(path_vtk))
    unstructuredGridToVTK(
            path_vtk,
            x, y, z,
            connectivity    = links,
            offsets         = offsets,
            cell_types      = cell_types,
            cellData        = tet_data,
            pointData       = point_data)

    return

def calculate_gravity(subdir_out, dir_matlab, name,  order):

    path_list = create_symlinks_gravity(subdir_out, name, order, path_list_only = True)
    files_exist = all([os.path.exists(path_) for path_ in path_list])

    if files_exist:

        print('All gravity files exist, skipping.')
        return

    # Record starting directory.
    start_dir = os.getcwd()

    try:

        os.chdir(dir_matlab)
        for anomaly_str in ['without_anomaly', 'with_anomaly']:

            command = 'matlab -nojvm -r "try; run_gravity(\'{:}\', \'model\', {:>1d}, \'{:}\'); catch e; fprintf(1, e.message); exit; end; exit"'.format(subdir_out, order, anomaly_str)
            print(command)
            os.system(command)

        #run_gravity('/Users/hrmd_work/Documents/research/stoneley/output/Magrathea/prem_0473.4_300/', 'model', 1, 'without_anomaly')
        create_symlinks_gravity(subdir_out, name, order, path_list_only = False)

    # If any goes wrong, go back to start directory.
    except:

        os.chdir(start_dir)
        raise

    return

# Main function. --------------------------------------------------------------
def test_build_ellipsoidal():

    mesh_size = 300.0
    r = 6371.0
    ellipticity = 1.0/10.0

    # Create the sphere mesh with embedded LLSVP outline.
    with pygmsh.geo.Geometry() as geom:

        # Create a representation of the ellipsoidal surface and add it 
        # to the geometry.
        ellipsoid_shell = create_ellipsoid_shell(geom, mesh_size, r, ellipticity)

        # Generate the mesh.
        mesh = geom.generate_mesh(dim = 2, verbose = False)

    # Remove construction points.
    pts, tri, i_mapping = get_real_points(mesh)
    pts = pts.T
    tri = tri.T

    # Unfortunately, it seems that Pygmsh doesn't guarantee that the points
    # fall on the surface of the sphere/ellipsoid, so here we readjust them.
    pts = rescale_to_surface(pts, r, ellipticity)
    
    print(pts.shape)
    x, y, z = pts
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    z_min = np.min(z)
    z_max = np.max(z)

    i_x_min = np.argmin(x)
    i_x_max = np.argmax(x)

    i_y_min = np.argmin(y)
    i_y_max = np.argmax(y)

    def circle(r):

        theta_span = np.linspace(0.0, 2.0*np.pi, num = 360)

        x = r*np.sin(theta_span)
        y = r*np.cos(theta_span)

        return x, y

    def ellipse(r, ell):

        theta = np.linspace(0.0, 2.0*np.pi, num = 360)
        cos_theta = np.cos(theta)
        r_of_theta = r*(1.0 - ((2.0/3.0)*ell*LegendrePoly2(cos_theta)))

        x = r_of_theta*np.sin(theta)
        y = r_of_theta*np.cos(theta)

        return x, y
    
    r_eq = (1.0 + (ellipticity/3.0))*r
    x_circle_eq, y_circle_eq = circle(r_eq)

    x_ell_eq, y_ell_eq = ellipse(r, ellipticity)

    import matplotlib.pyplot as plt
    fig, ax_arr = plt.subplots(1, 3, figsize = (11.0, 8.5))

    scatter_kwargs = {'s' : 1}

    ax = ax_arr[0]
    ax.scatter(x, y, **scatter_kwargs)
    ax.plot(x_circle_eq, y_circle_eq, c = 'r')
    for i in [i_x_min, i_x_max, i_y_min, i_y_max]:

        ax.scatter(x[i], y[i], c = 'g')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax = ax_arr[1]
    ax.scatter(x, z, **scatter_kwargs)
    ax.plot(x_ell_eq, y_ell_eq, c = 'r')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')

    ax = ax_arr[2]
    ax.scatter(y, z, **scatter_kwargs)
    ax.plot(x_ell_eq, y_ell_eq, c = 'r')
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')

    for ax in ax_arr:

        ax.set_aspect(1.0)

    plt.show()

    
    print('x_min:', ((-x_min/r) - 1.0)/ellipticity)
    print('x_max:', (( x_max/r) - 1.0)/ellipticity)
    print('y_min:', ((-y_min/r) - 1.0)/ellipticity)
    print('y_max:', (( y_max/r) - 1.0)/ellipticity)
    print('z_min:', ((-z_min/r) - 1.0)/ellipticity)
    print('z_max:', (( z_max/r) - 1.0)/ellipticity)
    
    return

def test_build_ellipsoidal2():
    
    #ellipticity = 1.0/300.0
    ellipticity = 3.334666966629331718e-03
    r = 6371.0

    #dir_output = '/Users/hrmd_work/Documents/research/stoneley/output/Magrathea/prem_0473.4_300/'
    dir_output = '/Users/hrmd_work/Documents/research/stoneley/output/Magrathea/prem_1019.8_300/'
    path_nodes = os.path.join(dir_output, 'model.1.node')

    nodes = np.loadtxt(path_nodes, skiprows = 1)

    _, x, y, z = nodes.T

    print(x.shape)

    from scipy.spatial import ConvexHull
    
    hull = ConvexHull(nodes[:, 1:])
    i_hull = hull.vertices

    print(i_hull[0:5])
    print(nodes.shape)

    x = x[i_hull]
    y = y[i_hull]
    z = z[i_hull]

    r_pts  = np.sqrt(x**2.0 + y**2.0 + z**2.0)

    print(np.min(r_pts))
    print(np.max(r_pts))

    r_h = np.sqrt(x**2.0 + y**2.0)

    theta = np.arctan2(r_h, z)
    cos_theta = np.cos(theta)

    r_ell = r*(1.0 - (2.0/3.0)*ellipticity*LegendrePoly2(cos_theta))

    print(np.max(np.abs(r_ell - r_pts)))


#    def circle(r):
#
#        theta_span = np.linspace(0.0, 2.0*np.pi, num = 360)
#
#        x = r*np.sin(theta_span)
#        y = r*np.cos(theta_span)
#
#        return x, y
#
#    def ellipse(r, ell):
#
#        theta = np.linspace(0.0, 2.0*np.pi, num = 360)
#        cos_theta = np.cos(theta)
#        r_of_theta = r*(1.0 - ((2.0/3.0)*ell*LegendrePoly2(cos_theta)))
#
#        x = r_of_theta*np.sin(theta)
#        y = r_of_theta*np.cos(theta)
#
#        return x, y
#    
#    r_eq = (1.0 + (ellipticity/3.0))*r
#    x_circle_eq, y_circle_eq = circle(r_eq)
#
#    x_ell_eq, y_ell_eq = ellipse(r, ellipticity)
#
#    import matplotlib.pyplot as plt
#    #fig, ax_arr = plt.subplots(1, 3, figsize = (11.0, 8.5))
#
#    scatter_kwargs = {'s' : 1}
#
#    fig = plt.figure(figsize = (11.0, 11.0))
#    ax = plt.gca()
#    ax.scatter(x, y, **scatter_kwargs)
#    ax.plot(x_circle_eq, y_circle_eq, c = 'r')
#    #for i in [i_x_min, i_x_max, i_y_min, i_y_max]:
#
#    #    ax.scatter(x[i], y[i], c = 'g')
#
#    ax.set_xlabel('X')
#    ax.set_ylabel('Y')
#    ax.set_aspect(1.0)
#
#    #ax = ax_arr[1]
#    fig = plt.figure(figsize = (11.0, 11.0))
#    ax = plt.gca()
#    ax.scatter(x, z, **scatter_kwargs)
#    ax.plot(x_ell_eq, y_ell_eq, c = 'r')
#    ax.set_xlabel('X')
#    ax.set_ylabel('Z')
#    ax.set_aspect(1.0)
#
#    #ax = ax_arr[2]
#    fig = plt.figure(figsize = (11.0, 11.0))
#    ax = plt.gca()
#    ax.scatter(y, z, **scatter_kwargs)
#    ax.plot(x_ell_eq, y_ell_eq, c = 'r')
#    ax.set_xlabel('Y')
#    ax.set_ylabel('Z')
#    ax.set_aspect(1.0)
#
#    ##for ax in ax_arr:
#
#    ##    ax.set_aspect(1.0)
#
#    plt.show()
    
    return

def main():

    # Find the input file.
    assert len(sys.argv) == 2, 'Usage: python3 build_ellipsoidal.py /path/to/input_file.txt'
    path_input = sys.argv[1]

    # Load input file.
    with open(path_input, 'r') as in_id:

        dir_input = in_id.readline().split()[1]
        dir_output = in_id.readline().split()[1]
        tet_max_vol = float(in_id.readline().split()[1])
        order = int(in_id.readline().split()[1])
        is_ellipsoidal = bool(int(in_id.readline().split()[1]))
        get_gravity = bool(int(in_id.readline().split()[1]))

    print('Read input file {:}'.format(path_input))
    print('Input directory: {:}'.format(dir_input))
    print('Output directory: {:}'.format(dir_output))
    print('Maximum tetrahedron volume: {:>.3e} km3'.format(tet_max_vol))
    print('Finite-element order: {:>1d}'.format(order))
    print('Model is spheroidal (not spherical): {:}'.format(is_ellipsoidal))
    print('Calculate gravity: {:}'.format(get_gravity))

    sys.exit()

    # Load model information.
    #file_model = 'prem_no_crust_03.0.txt'
    file_model = 'prem_no_80km_03.0.txt'
    path_model = os.path.join(dir_input, file_model)
    model, discon_lists = load_radial_model(path_model)
    
    # Load ellipticity information.
    if is_ellipsoidal:

        path_ellipticity = os.path.join(dir_input, 'ellipticity_profile.txt')
        ellipticity_data = np.loadtxt(path_ellipticity)
        r_ellipticity, ellipticity = ellipticity_data.T
        r_ellipticity = r_ellipticity*1.0E-3
        max_ellipticity = np.max(ellipticity)

    else:

        ellipticity_data = None
        max_ellipticity = 0.0

    if max_ellipticity == 0.0:

        ellipticity_str = 'sph'

    else:

        ellipticity_str = '{:>3d}'.format(int(round(1.0/max_ellipticity)))

    # Set the output directory.
    # The name includes the mesh size on the CMB ellipsoid
    # and the inverse of the ellipticity of the outer surface.
    mesh_size = 0.5*(2.0**(1.0/2.0))*(3.0**(1.0/3.0))*(tet_max_vol**(1.0/3.0))
    name_run = 'prem_{:>06.1f}_{:}'.format(mesh_size, ellipticity_str)
    subdir_out = os.path.join(dir_output, name_run)
    for dir_ in [dir_output, subdir_out]:
        mkdir_if_not_exist(dir_)

    # Choose the name of the model (prefixes the model files).
    name = 'model'

    # Set the largest allowed mesh sizes (km) on each discontinuity.
    #mesh_size_maxima = np.array([
    #     500.0, # ICB.
    #     500.0, # CMB.
    #     370.0, # Base of transition zone (670).
    #     180.0, # Top of transition zone (400).
    #     140.0, # Base of lithosphere.
    #      80.0, # Mid-lithosphere discontinuity.
    #      80.0  # Surface.
    #    ])
    mesh_size_maxima = np.array([
         500.0, # ICB.
         500.0, # CMB.
         270.0, # Base of transition zone (670).
         180.0, # Top of transition zone (400).
         220.0, # Base of lithosphere.
         220.0  # Surface.
        ])

    # Relax the mesh size limits (seems to give better results).
    mesh_size_maxima = 2.0*mesh_size_maxima

    # Make the .poly file, defining polygonal surfaces of regions.
    path_poly = make_ellipsoidal_poly_file_wrapper(dir_input, subdir_out, tet_max_vol, model, discon_lists, name, ellipticity_data, mesh_size_maxima)

    # Make the mesh sizing file (.b.poly), defining mesh size throughout domain.
    tet_min_max_edge_length_ratio = 2.0
    if ellipticity_data is None:

        raise NotImplementedError('Mesh sizing function not implemented for spherical case with multiple mantle discontinuities.')
        make_mesh_sizing_function_spherical(subdir_out, tet_max_vol, tet_min_max_edge_length_ratio, r_icb, r_cmb, r_srf, name)

    else:
        
        make_mesh_sizing_function_ellipsoidal(subdir_out, tet_max_vol, tet_min_max_edge_length_ratio, model, discon_lists, name, ellipticity_data, mesh_size_maxima)

    # Tetrahedralise the .poly file, creating the mesh.
    tetrahedralise_poly_file(tet_max_vol, path_poly, subdir_out, name)

    # Assign parameters at the mesh points.
    assign_parameters(dir_input, subdir_out, name, order, model, discon_lists, ellipticity_data)

    # Calculate the gravity field.
    if get_gravity:

        dir_matlab = os.path.join('.', 'matlab')
        calculate_gravity(subdir_out, dir_matlab, name, order)

    return

if __name__ == '__main__':

    main()
    #test_build_ellipsoidal()
    #test_build_ellipsoidal2()

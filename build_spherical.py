import os
import sys

import matplotlib.pyplot as plt
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

# Utilities. ------------------------------------------------------------------
def mkdir_if_not_exist(dir_):

    if not os.path.exists(dir_):

        os.mkdir(dir_)

    return

def rlonlat_to_xyz(r, lon, lat):
    '''
    Converts from radius, longitude and latitude to Cartesian coordinates.
    '''

    theta = (np.pi/2.0) - lat
        
    x = r*np.sin(theta)*np.cos(lon)
    y = r*np.sin(theta)*np.sin(lon)
    z = r*np.cos(theta)
    
    return x, y, z

def xyz_to_rlonlat(x, y, z):
    '''
    Converts from Cartesian coordinates to radius, longitude and latitude.
    '''

    r       = np.sqrt(x**2.0 + y**2.0 + z**2.0)
    theta   = np.arccos(z/r)
    lat     = (np.pi/2.0) - theta
    lon     = np.arctan2(y, x)

    return r, lon, lat

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

# Building the mesh. ----------------------------------------------------------
def make_spherical_poly_file_wrapper(dir_output, dir_input, subdir_out, tet_max_vol, r_icb, r_cmb, r_srf, name):

    # Determine the mesh size on the CMB sphere. 
    mesh_size = 0.5*(2.0**(1.0/2.0))*(3.0**(1.0/3.0))*(tet_max_vol**(1.0/3.0))
    #mesh_size = 3.0*mesh_size

    # Set output file for mesh points. 
    #subdir_out = os.path.join(dir_output, 'sphere_{:>07.2f}'.format(mesh_size))
    #
    out_path_pts_cmb = os.path.join(subdir_out, 'pts_cmb.npy')
    out_path_tri_cmb = os.path.join(subdir_out, 'tri_cmb.npy')
    out_path_i_pts_inside_llsvp_on_cmb = os.path.join(subdir_out, 'i_pts_inside_llsvp_on_cmb.npy')
    out_path_i_pts_boundary_llsvp_on_cmb = os.path.join(subdir_out, 'i_pts_boundary_llsvp_on_cmb.npy')
    #
    out_path_pts_srf = os.path.join(subdir_out, 'pts_srf.npy')
    out_path_tri_srf = os.path.join(subdir_out, 'tri_srf.npy')
    #
    out_path_pts_icb = os.path.join(subdir_out, 'pts_icb.npy')
    out_path_tri_icb = os.path.join(subdir_out, 'tri_icb.npy')

    # Load mesh points if already exist.
    path_list = [out_path_pts_cmb, out_path_tri_cmb,
        out_path_i_pts_inside_llsvp_on_cmb, out_path_i_pts_boundary_llsvp_on_cmb,
        out_path_pts_srf, out_path_tri_srf,
        out_path_pts_icb, out_path_tri_icb]
    files_exist = all([os.path.exists(path_) for path_ in path_list])

    if files_exist: 

        print('Points and triangulation files already exist.')
        print('Loading {:}.'.format(out_path_pts_cmb))
        pts_cmb = np.load(out_path_pts_cmb)
        print('Loading {:}.'.format(out_path_tri_cmb))
        tri_cmb = np.load(out_path_tri_cmb)
        print('Loading {:}.'.format(out_path_i_pts_inside_llsvp_on_cmb))
        i_pts_inside_llsvp_on_cmb = np.load(out_path_i_pts_inside_llsvp_on_cmb)
        print('Loading {:}.'.format(out_path_i_pts_boundary_llsvp_on_cmb))
        i_pts_boundary_llsvp_on_cmb = np.load(out_path_i_pts_boundary_llsvp_on_cmb)
        #
        print('Loading {:}.'.format(out_path_pts_srf))
        pts_srf = np.load(out_path_pts_srf)
        print('Loading {:}.'.format(out_path_tri_srf))
        tri_srf = np.load(out_path_tri_srf)
        #
        print('Loading {:}.'.format(out_path_pts_icb))
        pts_icb = np.load(out_path_pts_icb)
        print('Loading {:}.'.format(out_path_tri_icb))
        tri_icb = np.load(out_path_tri_icb)

    # Otherwise, create mesh points.
    else:

        pts_cmb, tri_cmb,                                           \
        i_pts_inside_llsvp_on_cmb, i_pts_boundary_llsvp_on_cmb  =   \
                build_cmb_sphere(subdir_out, dir_input, r_cmb, mesh_size)

        pts_srf, tri_srf = build_sphere(subdir_out, r_srf, mesh_size, 'srf')
        pts_icb, tri_icb = build_sphere(subdir_out, r_icb, mesh_size, 'icb')

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
    pts_llsvp_on_cmb = pts_cmb[:, i_pts_llsvp_on_cmb]
    pts_llsvp_top = move_points_outwards_spherical(pts_llsvp_on_cmb, d_r_llsvp)

    # Next, find all of the triangles in the bottom surface of the LLSVP.
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

    # Offset the top LLSVP point lists, because they will be listed after the
    # point lists for the CMB.
    n_pts_cmb = pts_cmb.shape[1]
    i_llsvp_top = i_llsvp_top + n_pts_cmb
    tri_llsvp_top = tri_llsvp_top + n_pts_cmb

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

    ## Plot.
    #fig = plt.figure()
    #ax = plt.gca()

    #r, lon, lat = xyz_to_rlonlat(*pts_llsvp_top)
    #lon_deg = np.rad2deg(lon)
    #lat_deg = np.rad2deg(lat)

    #ax.triplot(lon_deg, lat_deg, triangles = tri_cmb_top.T)

    #ax.set_xlabel('Longitude')
    #ax.set_ylabel('Latitude')
    #ax.set_xlim([-180.0, 180.0])
    #ax.set_ylim([-90.0, 90.0])
    #    
    #plt.show()

    # Apply offsets to ICB and surface sphere indices.
    n_pts_cmb_and_llsvp_top = n_pts_cmb + n_pts_llsvp_top
    n_pts_icb = pts_icb.shape[1]
    tri_icb = tri_icb + n_pts_cmb_and_llsvp_top
    tri_srf = tri_srf + n_pts_cmb_and_llsvp_top + n_pts_icb

    # Define the regions.
    # The polygonal surfaces define four regions (1 : inner core, 2 : outer core,
    # 3 : mantle, and 4 : LLSVP). These are identified in the .poly file by a single
    # point placed within them.
    # The points for the outer core and mantle are placed along the x-axis,
    # half-way between the bounding spheres.
    pt_inner_core   = [0.0,                   0.0, 0.0]
    pt_outer_core   = [(r_icb + r_cmb)/2.0,   0.0, 0.0]
    pt_mantle       = [(r_cmb + r_srf)/2.0,   0.0, 0.0]
    #
    # The point for the LLSVP is placed at the centroid of the outline,
    # halfway between the inner and outer surfaces.
    path_llsvp = os.path.join(dir_input, 'llsvp_smooth.txt')
    with open(path_llsvp, 'r') as in_id:
        
        _, llsvp_centre_lon_deg, llsvp_centre_lat_deg = in_id.readline().split()

    llsvp_centre_lon = np.deg2rad(float(llsvp_centre_lon_deg))
    llsvp_centre_lat = np.deg2rad(float(llsvp_centre_lat_deg))
    llsvp_centre_r = r_cmb + (d_r_llsvp/2.0) 
    pt_centre_llsvp = list(rlonlat_to_xyz(llsvp_centre_r, llsvp_centre_lon, llsvp_centre_lat))
    #
    pts_regions = np.array([pt_inner_core, pt_outer_core, pt_mantle, pt_centre_llsvp]).T

    # Write the .poly file.
    path_poly, path_tetgen_ele = make_spherical_poly_file(pts_cmb, pts_llsvp_top, pts_icb, pts_srf, tri_cmb, tri_llsvp_top, quad_llsvp_sides, tri_icb, tri_srf, pts_regions, subdir_out, name)

    return path_poly, path_tetgen_ele

def build_sphere(subdir_out, r, mesh_size, name = None):

    # Create the sphere mesh with embedded LLSVP outline.
    with pygmsh.geo.Geometry() as geom:

        # Add the ball (spherical surface).
        ball = geom.add_ball([0.0, 0.0, 0.0], r, mesh_size = mesh_size)

        # Generate the mesh.
        mesh = geom.generate_mesh(dim = 2, verbose = False)

    # Remove construction points.
    pts, tri, i_mapping = get_real_points(mesh)
    pts = pts.T
    tri = tri.T

    # Unfortunately, it seems that Pygmsh doesn't guarantee that the points
    # fall on the surface of the sphere, so here we readjust them.
    r_pts = np.linalg.norm(pts, axis = 0)
    pts = r*pts/r_pts

    #r_pts = np.linalg.norm(pts, axis = 1)
    #print(np.min(r_pts), np.max(r_pts))

    if name is not None:

        # Save points and triangulation.
        # First create output directory.
        #subdir_out = os.path.join(dir_output, 'sphere_{:>07.2f}'.format(mesh_size))
        #for dir_ in [dir_output, subdir_out]:
        #    mkdir_if_not_exist(dir_)
        #mkdir_if_not_exist(subdir_out)

        # Second, save points.
        out_path_pts = os.path.join(subdir_out, 'pts_{:}.npy'.format(name))
        print('Saving points to {:}'.format(out_path_pts))
        np.save(out_path_pts, pts)

        # Third, save triangulation.
        out_path_tri = os.path.join(subdir_out, 'tri_{:}.npy'.format(name))
        print('Saving triangulation to {:}'.format(out_path_tri))
        np.save(out_path_tri, tri)

    return pts, tri

def build_cmb_sphere(subdir_out, dir_input, r_cmb, mesh_size):

    # Define a mesh coarsening factor.
    # Seems necessary to coarsen the mesh near the LLSVP boundary to maintain
    # a relatively even mesh size.
    mesh_size_factor = 2.0 

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
    x_pts, y_pts, z_pts = rlonlat_to_xyz(r_cmb, lon_pts_rads, lat_pts_rads) 
    pts = np.array([x_pts, y_pts, z_pts]).T
    n_pts = pts.shape[0]

    # Create the sphere mesh with embedded LLSVP outline.
    with pygmsh.geo.Geometry() as geom:

        # Add the ball (spherical surface).
        ball_srf = geom.add_ball([0.0, 0.0, 0.0], r_cmb, mesh_size = mesh_size)

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

            geom.in_surface(line, ball_srf.surface_loop)

            line_list.append(line)

        #print(line_list)
        curve_loop = geom.add_curve_loop(line_list)
        #geom.add_physical(curve_loop, label = "outline")
        #for i in range(n_pts):
        #    geom.add_physical(line_list[i], label = 'line_segment_{:>09d}'.format(i))

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
    # fall on the surface of the sphere, so here we readjust them.
    r_pts = np.linalg.norm(pts, axis = 0)
    pts = r_cmb*pts/r_pts

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
    _, lon, lat = xyz_to_rlonlat(x, y, z)
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

    # Remove longitude shift.
    lon_deg = lon_deg - shift
    i_2pi_shift = (lon_deg < -180.0)
    lon_deg[i_2pi_shift] = lon_deg[i_2pi_shift] + 360.0

    # Convert back to x, y and z.
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    x, y, z = rlonlat_to_xyz(r_cmb, lon_rad, lat_rad)

    # Save points and triangulation.
    # First create output directory.
    #subdir_out = os.path.join(dir_output, 'sphere_{:>07.2f}'.format(mesh_size))
    #for dir_ in [dir_output, subdir_out]:
    #    mkdir_if_not_exist(dir_)
    #mkdir_if_not_exist(subdir_out)

    # Second, save points.
    pts = np.array([x, y, z])
    out_path_pts = os.path.join(subdir_out, 'pts_cmb.npy')
    print('Saving points to {:}'.format(out_path_pts))
    np.save(out_path_pts, pts)

    # Third, save triangulation.
    out_path_tri = os.path.join(subdir_out, 'tri_cmb.npy')
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

        # Plot.
        fig = plt.figure()
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
        
        #plt.show()

        fig = plt.figure(figsize = (8.5, 8.5))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_trisurf(x, y, z, triangles = tri.T, edgecolor = (1.0, 0.0, 0.0, 0.2), color = (1.0, 1.0, 1.0, 1.0)) #cmap = 'magma') 
        ax.plot(x[i_outline], y[i_outline], z[i_outline], zorder = 10)
        ax.scatter(x[i_inside], y[i_inside], z[i_inside], color = 'g', zorder = 11)

        plt.show()

    return pts, tri, i_inside, i_outline

def move_points_outwards_spherical(pts, d_r):

    # Convert to geographical coordinates.
    r, lon, lat = xyz_to_rlonlat(*pts)

    # Add the outward offset.
    r = r + d_r

    # Convert back to Cartesian coordinates.
    x, y, z = rlonlat_to_xyz(r, lon, lat)
    pts = np.array([x, y, z])

    return pts

def make_mesh_sizing_function(subdir_out, tet_max_vol, tet_min_max_edge_length_ratio, r_icb, r_cmb, r_srf, name):

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

def make_spherical_poly_file(pts_cmb, pts_llsvp_top, pts_icb, pts_srf, tri_cmb, tri_llsvp_top, quad_llsvp_sides, tri_icb, tri_srf, pts_regions, subdir_out, name):
    '''
    See section 5.2.2 of the TetGen manual for the specification of the .poly
    file.
    '''

    # Define output path.
    #name = 'spheres'
    path_poly = os.path.join(subdir_out, '{:}.poly'.format(name))
    path_tetgen_ele = os.path.join(subdir_out, '{:}.1.ele'.format(name))

    # Merge point lists.
    pts = np.concatenate([pts_cmb, pts_llsvp_top, pts_icb, pts_srf], axis = 1)

    # Get information about mesh.
    n_dim = pts.shape[0]
    n_nodes = pts.shape[1]
    #n_tri = tri.shape[1]
    n_tri_cmb = tri_cmb.shape[1]
    n_tri_llsvp_top = tri_llsvp_top.shape[1]
    n_quad_llsvp_sides = quad_llsvp_sides.shape[1]
    n_tri_icb = tri_icb.shape[1]
    n_tri_srf = tri_srf.shape[1]
    n_facets = n_tri_cmb + n_tri_llsvp_top + n_quad_llsvp_sides + n_tri_icb + n_tri_srf
    #
    n_regions = pts_regions.shape[1]

    # Shift from 0-based to 1-based indexing.
    tri_cmb = tri_cmb + 1
    tri_llsvp_top = tri_llsvp_top + 1
    quad_llsvp_sides = quad_llsvp_sides + 1
    tri_icb = tri_icb + 1
    tri_srf = tri_srf + 1
    
    # Header lines for part one: Node list.
    comments_1a = ['# Part 1 - node list', '# node count, 3 dim, no attribute, no boundary marker']
    comment_1b = '# Node index, node coordinates'
    header_1 = '{:>8d}{:>8d}{:>8d}{:>8d}'.format(n_nodes, n_dim, 0, 0)
    header_and_comments_1 = [*comments_1a, header_1, comment_1b]

    # Header lines for part two: Facet list.
    comments_2a = ['# Part 2 - facet list', '# facet count, no boundary marker']
    #header_2 = '{:>8d}{:>8d}'.format(n_tri, 0)
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

        # Write the node list for part 1.
        for i in range(n_nodes):

            # Note: Write (i + 1) instead of (i) because of 1-based indexing.
            out_id.write('{:>8d} {:>+20.14e} {:>+20.14e} {:>+20.14e}\n'.format(i + 1, pts[0, i], pts[1, i], pts[2, i]))

        # Blank line before part 2.
        out_id.write('\n')

        # Write header lines for part 2.
        for line in header_and_comments_2:

            out_id.write('{:}\n'.format(line))

        # Write the facet list for part 2.

        # First: Facets of the CMB.
        # Comment line.
        out_id.write('# Facets (triangles) of CMB.\n')
        #
        for i in range(n_tri_cmb):

            # Write '1' to indicate each facet is a single triangle.
            out_id.write('{:>8d}\n'.format(1))

            # Write 3, i0, i1, i2 where 3 is the number of vertices of the triangle,
            # and i0, i1, i2 are the indices of the vertices of the triangle.
            out_id.write('{:>8d} {:>8d} {:>8d} {:>8d}\n'.format(3, tri_cmb[0, i], tri_cmb[1, i], tri_cmb[2, i]))

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

        # Fourth: Facets of the ICB.
        # Comment line.
        out_id.write('# Facets (triangles) of ICB.\n')
        #
        for i in range(n_tri_icb):

            # Write '1' to indicate each facet is a single triangle.
            out_id.write('{:>8d}\n'.format(1))

            # Write 3, i0, i1, i2 where 3 is the number of vertices of the triangle,
            # and i0, i1, i2 are the indices of the vertices of the triangle.
            out_id.write('{:>8d} {:>8d} {:>8d} {:>8d}\n'.format(3, tri_icb[0, i], tri_icb[1, i], tri_icb[2, i]))

        # Fifth: Facets of the surface.
        # Comment line.
        out_id.write('# Facets (triangles) of outer surface.\n')
        #
        for i in range(n_tri_srf):

            # Write '1' to indicate each facet is a single triangle.
            out_id.write('{:>8d}\n'.format(1))

            # Write 3, i0, i1, i2 where 3 is the number of vertices of the triangle,
            # and i0, i1, i2 are the indices of the vertices of the triangle.
            out_id.write('{:>8d} {:>8d} {:>8d} {:>8d}\n'.format(3, tri_srf[0, i], tri_srf[1, i], tri_srf[2, i]))

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

    return path_poly, path_tetgen_ele

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

    #command = 'tetgen -A -pmq{:.2f}/{:.2f}nYCVFO5/7 -a {:>7.2f} {:}'.format(radius_edge_ratio, max_dihedral_angle, tet_max_vol, path_poly)
    command = 'tetgen -k -A -pmq{:.2f}/{:.2f}nYCVFO5/7 -a {:>7.2f} {:}'.format(radius_edge_ratio, max_dihedral_angle, tet_max_vol, path_poly)
    print(command)
    os.system(command)

    return 

# Assigning parameters at mesh points. ----------------------------------------
def assign_parameters(dir_input, subdir_out, name, order):

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

    # Create the output arrays.
    n_tets = tets.shape[1]
    n_nodes_per_tet = nodal_pts.shape[1]
    #
    v_p = np.zeros((n_nodes_per_tet, n_tets))
    v_s = np.zeros((n_nodes_per_tet, n_tets))
    rho = np.zeros((n_nodes_per_tet, n_tets))
    
    # Load the input model.
    r_mod, v_p_mod, v_s_mod, rho_mod, i_icb, i_cmb, i_inner_core,       \
    i_outer_core, i_mantle =                                            \
            load_radial_model(dir_input)

    # Interpolate the radial model at the nodal points.
    label_list = [1, 2, 3, 4]
    i_list = [i_inner_core, i_outer_core, i_mantle, i_mantle]
    for j in range(4):

        # Find the tetrahedron in the specified region.
        label = label_list[j]
        k = np.where(tet_labels == label)[0]
        nodal_pts_region = nodal_pts[:, :, k]
        r_pts_region = np.linalg.norm(nodal_pts_region, axis = 0)

        # Extract the relevant portion of the reference model.
        r_mod_region    = r_mod[i_list[j]]
        v_p_mod_region  = v_p_mod[i_list[j]]
        v_s_mod_region  = v_s_mod[i_list[j]]
        rho_mod_region  = rho_mod[i_list[j]]

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
    k = np.where(tet_labels == 4)[0]
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
    path_vtk = os.path.join(subdir_out, '{:}'.format(name))
    save_model_to_vtk(path_vtk, nodes, tets, links, tet_labels, v_p, v_s, rho)

    # Create symbolic links.
    dir_with_anomaly = os.path.join(subdir_out, 'with_anomaly')
    dir_without_anomaly = os.path.join(subdir_out, 'without_anomaly')
    for dir_ in [dir_with_anomaly, dir_without_anomaly]:

        mkdir_if_not_exist(dir_)

    current_dir = os.getcwd()


    os.chdir(dir_with_anomaly)

    rel_path_node_dat       = os.path.join('..', file_node_dat)
    rel_path_ele_dat        = os.path.join('..', file_ele_dat)
    rel_path_neigh_dat      = os.path.join('..', file_neigh_dat)
    rel_path_mesh_header    = os.path.join('..', file_mesh_header)

    path_anomaly_symmesh_header        = os.path.join('..', file_mesh_header)
    path_anomaly_symlink_node_dat   = os.path.join('..', file_node_dat)
    path_anomaly_symlink_ele_dat    = os.path.join('..', file_ele_dat)
    path_anomaly_symlink_neigh_dat  = os.path.join('..', file_neigh_dat)
    #
    file_symlink_rho_dat = '{:}.1_rho_pod_{:1d}_true.dat'.format(name, order)
    file_symlink_v_p_dat = '{:}.1_v_p_pod_{:1d}_true.dat'.format(name, order)
    file_symlink_v_s_dat = '{:}.1_v_s_pod_{:1d}_true.dat'.format(name, order)
    #path_anomaly_symlink_rho_dat = os.path.join('..', file_symlink_rho_dat)
    #path_anomaly_symlink_v_p_dat = os.path.join('..', file_symlink_v_p_dat)
    #path_anomaly_symlink_v_s_dat = os.path.join('..', file_symlink_v_s_dat)
    #
    file_anomaly_rho_dat = '{:}.1_rho_pod_{:1d}_true_with_anomaly.dat'.format(name, order)
    file_anomaly_v_p_dat = '{:}.1_v_p_pod_{:1d}_true_with_anomaly.dat'.format(name, order)
    file_anomaly_v_s_dat = '{:}.1_v_s_pod_{:1d}_true_with_anomaly.dat'.format(name, order)
    #
    rel_path_anomaly_rho_dat = os.path.join('..', file_anomaly_rho_dat)
    rel_path_anomaly_v_p_dat = os.path.join('..', file_anomaly_v_p_dat)
    rel_path_anomaly_v_s_dat = os.path.join('..', file_anomaly_v_s_dat)
    #
    file_no_anomaly_rho_dat = '{:}.1_rho_pod_{:1d}_true_without_anomaly.dat'.format(name, order)
    file_no_anomaly_v_p_dat = '{:}.1_v_p_pod_{:1d}_true_without_anomaly.dat'.format(name, order)
    file_no_anomaly_v_s_dat = '{:}.1_v_s_pod_{:1d}_true_without_anomaly.dat'.format(name, order)
    #
    rel_path_no_anomaly_rho_dat = os.path.join('..', file_no_anomaly_rho_dat)
    rel_path_no_anomaly_v_p_dat = os.path.join('..', file_no_anomaly_v_p_dat)
    rel_path_no_anomaly_v_s_dat = os.path.join('..', file_no_anomaly_v_s_dat)

    path_pairs_with_anomaly =\
                [   [rel_path_node_dat,             file_node_dat], 
                    [rel_path_ele_dat,              file_ele_dat], 
                    [rel_path_neigh_dat,            file_neigh_dat],
                    [rel_path_mesh_header,          file_mesh_header],
                    [rel_path_anomaly_rho_dat,      file_symlink_rho_dat],
                    [rel_path_anomaly_v_p_dat,      file_symlink_v_p_dat], 
                    [rel_path_anomaly_v_s_dat,      file_symlink_v_s_dat]]

    path_pairs_without_anomaly =\
                [   [rel_path_node_dat,                 file_node_dat], 
                    [rel_path_ele_dat,                  file_ele_dat], 
                    [rel_path_neigh_dat,                file_neigh_dat],
                    [rel_path_mesh_header,              file_mesh_header],
                    [rel_path_no_anomaly_rho_dat,      file_symlink_rho_dat],
                    [rel_path_no_anomaly_v_p_dat,      file_symlink_v_p_dat], 
                    [rel_path_no_anomaly_v_s_dat,      file_symlink_v_s_dat]]

    for path_pair in path_pairs_with_anomaly:

        path_true = path_pair[0]
        path_symlink = path_pair[1]

        command = 'ln -sf {:} {:}'.format(path_true, path_symlink)
        print(command)
        os.system(command)

    os.chdir(dir_without_anomaly)

    for path_pair in path_pairs_without_anomaly:

        path_true = path_pair[0]
        path_symlink = path_pair[1]

        command = 'ln -sf {:} {:}'.format(path_true, path_symlink)
        print(command)
        os.system(command)

    os.chdir(current_dir)

    return

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

def load_radial_model(dir_input, name = 'prem_noocean'):

    # Load the model.
    path_model = os.path.join(dir_input, '{:}.txt'.format(name))
    data = np.loadtxt(path_model, skiprows = 3)

    # Unpack the data (all in SI units).
    r       = data[:, 0] # Radial coordinate (m).
    rho     = data[:, 1] # Density (kg/m3).
    v_pv    = data[:, 2] # Vertically-polarised P-wave speed (m/s).
    v_sv    = data[:, 3] # Vertically-polarised S-wave speed (m/s).
    q_ka    = data[:, 4] # Q-factor for kappa.
    q_mu    = data[:, 5] # Q-factor for mu.
    v_ph    = data[:, 6] # Horizontally-polarised P-wave speed (m/s).
    v_sh    = data[:, 7] # Horizontally-polarised S-wave speed (m/s).
    eta     = data[:, 8] # Reference frequency (s).

    # Find the isotropic averages of the anisotropic speeds.
    v_p = (v_pv + v_ph)/2.0
    v_s = (v_sv + v_sv)/2.0

    # Convert to units used in NormalModes.
    r = r*1.0E-3        # m to km.
    v_p = v_p*1.0E-3    # m/s to km/s.
    v_s = v_s*1.0E-3    # m/s to km/s.
    rho = rho*1.0E-3    # kg/m3 to g/cm3.

    # Find the indices of the fluid outer core.
    i_fluid = np.where(v_s < 1.0E-11)[0]
    i_cmb = i_fluid[-1] + 1 # Index of lowermost layer in mantle.
    i_icb = i_fluid[0] # Index of lowermost layer in outer core.
    #
    n_layers = len(r)
    i_inner_core = np.array(list(range(0, i_icb)), dtype = np.int)
    i_outer_core = np.array(list(range(i_icb, i_cmb)), dtype = np.int)
    i_mantle = np.array(list(range(i_cmb, n_layers)), dtype = np.int)

    return r, v_p, v_s, rho, i_icb, i_cmb, i_inner_core, i_outer_core, i_mantle

def save_model_to_vtk(path_vtk, pts, tets, links, tet_labels, v_p, v_s, rho):
    
    # Get number of tetrahedra.
    n_tets = tets.shape[1]

    # Define offsets (in the flattened tetrahedra connectivity list, the
    # index of the last node of each tetrahedron). Node 1-based indexing
    # is used for this list.
    offsets = 4*(np.array(list(range(n_tets)), dtype = np.int) + 1)
    
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
    cell_types = np.zeros(n_tets, dtype = np.int) + VtkTetra.tid

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

# Main function. --------------------------------------------------------------
def main():

    # Find the input file.
    assert len(sys.argv) == 2, 'Usage: python3 build_spherical.py /path/to/input_file.txt'
    path_input = sys.argv[1]

    # Load input file.
    with open(path_input, 'r') as in_id:

        dir_input = in_id.readline().strip()
        dir_output = in_id.readline().strip()
        tet_max_vol = float(in_id.readline())
        order = int(in_id.readline())
    
    # Load radius information.
    path_input = os.path.join(dir_input, 'radii.txt')
    with open(path_input, 'r') as in_id:

        r_srf = float(in_id.readline())
        r_cmb = float(in_id.readline())
        r_icb = float(in_id.readline())

    # Set the output directory.
    # The name includes the mesh size on the CMB sphere. 
    mesh_size = 0.5*(2.0**(1.0/2.0))*(3.0**(1.0/3.0))*(tet_max_vol**(1.0/3.0))
    subdir_out = os.path.join(dir_output, 'sphere_{:>06.1f}'.format(mesh_size))
    for dir_ in [dir_output, subdir_out]:
        mkdir_if_not_exist(dir_)

    name = 'spheres'

    # Make the .poly file, defining polygonal surfaces of regions.
    path_poly, path_tetgen_ele = make_spherical_poly_file_wrapper(dir_output, dir_input, subdir_out, tet_max_vol, r_icb, r_cmb, r_srf, name)

    # Make the mesh sizing file (.b.poly), defining mesh size throughout domain.
    tet_min_max_edge_length_ratio = 2.0
    make_mesh_sizing_function(subdir_out, tet_max_vol, tet_min_max_edge_length_ratio, r_icb, r_cmb, r_srf, name)

    # Tetrahedralise the .poly file, creating the mesh.
    tetrahedralise_poly_file(tet_max_vol, path_poly, subdir_out, name)

    # Assign parameters at the mesh points.
    assign_parameters(dir_input, subdir_out, name, order)

    return

if __name__ == '__main__':

    main()

# Old. ------------------------------------------------------------------------

def make_spherical_poly_file_old(pts, tri, subdir_out):
    '''
    See section 5.2.2 of the TetGen manual for the specification of the .poly
    file.
    '''

    # Define output path.
    name = 'spheres'
    path_poly = os.path.join(subdir_out, '{:}.poly'.format(name))
    path_tetgen_ele = os.path.join(subdir_out, '{:}.1.ele'.format(name))

    # Get information about mesh.
    n_dim = pts.shape[0]
    n_nodes = pts.shape[1]
    n_tri = tri.shape[1]

    # Shift from 0-based to 1-based indexing.
    tri = tri + 1
    
    # Header lines for part one: Node list.
    comments_1a = ['# Part 1 - node list', '# node count, 3 dim, no attribute, no boundary marker']
    comment_1b = '# Node index, node coordinates'
    header_1 = '{:>8d}{:>8d}{:>8d}{:>8d}'.format(n_nodes, n_dim, 0, 0)
    header_and_comments_1 = [*comments_1a, header_1, comment_1b]

    # Header lines for part two: Facet list.
    comments_2a = ['# Part 2 - facet list', '# facet count, no boundary marker']
    header_2 = '{:>8d}{:>8d}'.format(n_tri, 0)
    comment_2b = '# facets'
    header_and_comments_2 = [*comments_2a, header_2, comment_2b]

    # Lines for parts 3 and 4 (hole and region lists).
    comment_3 = '# Part 3 - hole list'
    header_3 = '{:>8d}'.format(0)
    #
    comment_4 = '# Part 4 - region list'
    header_4 = '{:>8d}'.format(0)
    #
    # Note blank line between 3 and 4.
    header_and_comments_3_and_4 = [comment_3, header_3, '', comment_4, header_4]

    # Write the file.
    print('Writing .poly file {:}'.format(path_poly))
    with open(path_poly, 'w') as out_id:

        # Write header lines for part 1.
        for line in header_and_comments_1: 

            out_id.write('{:}\n'.format(line))

        # Write the node list for part 1.
        for i in range(n_nodes):

            # Note: Write (i + 1) instead of (i) because of 1-based indexing.
            out_id.write('{:>8d} {:>+20.14e} {:>+20.14e} {:>+20.14e}\n'.format(i + 1, pts[0, i], pts[1, i], pts[2, i]))

        # Blank line before part 2.
        out_id.write('\n')

        # Write header lines for part 2.
        for line in header_and_comments_2:

            out_id.write('{:}\n'.format(line))

        # Write the facet list for part 2.
        for i in range(n_tri):

            # Write '1' to indicate each facet is a single triangle.
            out_id.write('{:>8d}\n'.format(1))

            # Write 3, i0, i1, i2 where 3 is the number of vertices of the triangle,
            # and i0, i1, i2 are the indices of the vertices of the triangle.
            out_id.write('{:>8d} {:>8d} {:>8d} {:>8d}\n'.format(3, tri[0, i], tri[1, i], tri[2, i]))

        # Blank line before parts 3 and 4.
        out_id.write('\n')

        # Write parts 3 and 4.
        for line in header_and_comments_3_and_4:

            out_id.write('{:}\n'.format(line))

    return path_poly, path_tetgen_ele

def convert_tetgen_to_vtk(path_tetgen_ele):

    dir_name = os.path.dirname(path_tetgen_ele)
    base_name = '.'.join(os.path.basename(path_tetgen_ele).split('.')[:-1])
    path_vtk = os.path.join(dir_name, '{:}.vtk'.format(base_name)) 

    print('Reading TetGen mesh: {:}'.format(path_tetgen_ele))
    mesh = meshio.read(path_tetgen_ele)
    print('Writing VTK mesh: {:}'.format(path_vtk))
    mesh.write(path_vtk)

    return

def old():

    #tets_copy  = tets.copy()
    #tets[0, :] = tets_copy[0, :]
    #tets[1, :] = tets_copy[1, :]
    #tets[2, :] = tets_copy[3, :]
    #tets[3, :] = tets_copy[2, :]

    #tets[[2,3], :] = tets[[3, 2], :]
    #v_p[[2,3], :] = v_p[[3, 2], :]
    #v_s[[2,3], :] = v_s[[3, 2], :]
    #rho[[2,3], :] = rho[[3, 2], :]

    #n_tets = tets.shape[1]
    #polarity = np.zeros(n_tets, dtype = np.int)

    #for i in range(n_tets):

    #    tet = tets[:, i]
    #    i0 = tet[0]
    #    i1 = tet[1]
    #    i2 = tet[2]
    #    i3 = tet[3]

    #    v01 = pts[:, i1] - pts[:, i0]
    #    v02 = pts[:, i2] - pts[:, i0]
    #    v03 = pts[:, i3] - pts[:, i0]

    #    v01_cross_v02 = np.cross(v01, v02)
    #    polarity[i] = int(np.sign(np.dot(v01_cross_v02, v03)))

    #print(np.sum(polarity == -1))
    #print(np.sum(polarity ==  0))
    #print(np.sum(polarity ==  1))

    return

def old_save_model_to_vtk(path_vtk):
    '''
    http://spikard.blogspot.com/2015/07/visualization-of-unstructured-grid-data.html
    '''
    #def dump_vtu_evtk(filename, x, y, faces, ux, uy):
    # vec_speed = (ux, uy, np.zeros_like(ux))
    # vertices = (x, y, np.zeros_like(x))
    # w = VtkFile(filename, VtkUnstructuredGrid)
    # w.openGrid()
    # w.openPiece(npoints=len(x), ncells=len(faces))
    # w.openData("Cell", vectors="Velocity")
    # w.addData("Velocity", vec_speed)
    # w.closeData("Cell")
    # w.openElement("Points")
    # w.addData("Points", vertices)
    # w.closeElement("Points")
    # ncells = len(faces)
    # # index of last node in each cell
    # offsets = np.arange(start=3, stop=3*(ncells + 1), step=3,
    #                     dtype='uint32')
    # print offsets, len(offsets)
    # connectivity = faces.reshape(ncells*3).astype('int32') + 1
    # print connectivity, len(connectivity)
    # cell_types = np.empty(ncells, dtype='uint8')
    # cell_types[:] = VtkTriangle.tid
    # print cell_types, len(cell_types)

    vtk_file = evtk.vtk.VtkFile(path_vtk, evtk.vtk.VtkUnstructuredGrid)
    print(vtk_file)

    return

def old_save_model_to_vtk2(path_vtk):

    # Define vertices
    x = np.zeros(6)
    y = np.zeros(6)
    z = np.zeros(6)
    
    x[0], y[0], z[0] = 0.0, 0.0, 0.0
    x[1], y[1], z[1] = 1.0, 0.0, 0.0
    x[2], y[2], z[2] = 2.0, 0.0, 0.0
    x[3], y[3], z[3] = 0.0, 1.0, 0.0
    x[4], y[4], z[4] = 1.0, 1.0, 0.0
    x[5], y[5], z[5] = 2.0, 1.0, 0.0
    
    #point_data = {"test_pd": np.array([1, 2, 3, 4, 5, 6])}
    point_data = {"test_pd": np.array([0, 0, 0, 1, 1, 1])}
    cell_data = {"test_cd": np.array([1, 2, 3])}
    field_data = {"test_fd": np.array([1.0, 2.0])}
    # Define connectivity or vertices that belongs to each element

    # Define connectivity or vertices that belongs to each element
    conn = np.zeros(10)
    
    conn[0], conn[1], conn[2] = 0, 1, 3              # first triangle
    conn[3], conn[4], conn[5] = 1, 4, 3              # second triangle
    conn[6], conn[7], conn[8], conn[9] = 1, 2, 5, 4  # rectangle
    
    # Define offset of last vertex of each element
    offset = np.zeros(3)
    offset[0] = 3
    offset[1] = 6
    offset[2] = 10
    
    # Define cell types
    
    ctype = np.zeros(3)
    ctype[0], ctype[1] = VtkTriangle.tid, VtkTriangle.tid
    ctype[2] = VtkQuad.tid
    
    comments = [ "comment 1", "comment 2" ]
    unstructuredGridToVTK("unstructured", x, y, z, connectivity = conn, offsets = offset, cell_types = ctype, cellData = cell_data, pointData = point_data)

    return

def old_save_model_to_vtk4(path_vtk):

    # Define vertices
    x = np.array([0.0, 1.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0, 1.0])
    
    #point_data = {"test_pd": np.array([1, 2, 3, 4, 5, 6])}
    #cell_data = {"test_cd": np.array([1, 2, 3])}
    #field_data = {"test_fd": np.array([1.0, 2.0])}
    # Define connectivity or vertices that belongs to each element

    ## Define connectivity or vertices that belongs to each element
    #conn = np.zeros(10)
    #
    #conn[0], conn[1], conn[2] = 0, 1, 3              # first triangle
    #conn[3], conn[4], conn[5] = 1, 4, 3              # second triangle
    #conn[6], conn[7], conn[8], conn[9] = 1, 2, 5, 4  # rectangle

    
    # Define offset of last vertex of each element
    offset = np.zeros(3)
    offset[0] = 3
    offset[1] = 6
    offset[2] = 10
    
    # Define cell types
    
    ctype = np.zeros(3)
    ctype[0], ctype[1] = VtkTriangle.tid, VtkTriangle.tid
    ctype[2] = VtkQuad.tid
    
    comments = [ "comment 1", "comment 2" ]
    unstructuredGridToVTK("unstructured", x, y, z, connectivity = conn, offsets = offset, cell_types = ctype, cellData = cell_data, pointData = point_data)

    return


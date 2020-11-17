import os

import matplotlib.pyplot as plt
import numpy as np

from build_spherical import build_ellipsoid, build_cmb_ellipsoid

def get_mean_edge_length(r, n_tri):

    surface_area = 4.0*np.pi*(r**2.0)
    mean_area_tri = surface_area/n_tri
    equivalent_mean_edge_length = np.sqrt((4.0*np.sqrt(3.0)*mean_area_tri)/3.0)

    return equivalent_mean_edge_length

def main():
    
    file_test_tri = 'tri_test_data.npy'
    path_test_tri = os.path.join('../../output/Magrathea/', file_test_tri)
    if os.path.exists(path_test_tri):

        data = np.load(path_test_tri, allow_pickle = True).item()

    else:

        dir_input = '../../input/Magrathea/llsvp/'

        r = 6371.0
        mesh_size = 500.0
        ellipticity = 1.0/300.0

        mesh_size = [300.0, 400.0, 500.0]
        #mesh_size = [300.0]
        #mesh_size = [100.0, 200.0, 300.0, 400.0, 500.0]
        n_mesh_size = len(mesh_size)

        n_tri_plain = np.zeros(n_mesh_size, dtype = np.int)
        n_tri_llsvp = np.zeros(n_mesh_size, dtype = np.int)

        for i in range(n_mesh_size):

            print('Building plain triangulation with mesh size {:.1f} km'.format(mesh_size[i]))

            pts, tri = build_ellipsoid(None, r, mesh_size[i], None)
            n_tri_plain[i] = tri.shape[1]

        for i in range(n_mesh_size):

            print('Building LLSVP triangulation with mesh size {:.1f} km'.format(mesh_size[i]))

            pts, tri, _, _ = build_cmb_ellipsoid(None, dir_input, r, mesh_size[i], ellipticity, None, make_plots = True)
            n_tri_llsvp[i] = tri.shape[1]

        data = dict()
        data['r'] = r
        data['mesh_size'] = mesh_size
        data['n_tri_plain'] = n_tri_plain
        data['n_tri_llsvp'] = n_tri_llsvp

        np.save(path_test_tri, data)

    r = data['r']
    mesh_size = data['mesh_size']
    n_tri_plain = data['n_tri_plain']
    n_tri_llsvp = data['n_tri_llsvp']

    equivalent_mean_edge_length_plain = get_mean_edge_length(r, n_tri_plain)
    equivalent_mean_edge_length_llsvp = get_mean_edge_length(r, n_tri_llsvp)

    mesh_size_min = np.min(mesh_size)
    mesh_size_max = np.max(mesh_size)
    mesh_size_range = mesh_size_max - mesh_size_min
    buff = 0.05
    mesh_size_lim_min = mesh_size_min - buff*mesh_size_range
    mesh_size_lim_max = mesh_size_max + buff*mesh_size_range
    mesh_size_lims = [mesh_size_lim_min, mesh_size_lim_max]
    
    line_kwargs = {'marker' : '.', 'linestyle' : '-'}
    fig = plt.figure(figsize = (8.5, 8.5))
    ax = plt.gca()

    #plt.plot(mesh_size, n_tri, marker = '.', linestyle = '-', color = 'k')
    plt.plot(mesh_size, equivalent_mean_edge_length_llsvp, **line_kwargs, color = 'g', label = 'LLSVP')
    plt.plot(mesh_size, equivalent_mean_edge_length_plain, **line_kwargs, color = 'b', label = 'Plain')

    plt.plot(mesh_size_lims, mesh_size_lims, linestyle = ':', color = 'k')

    ax.legend(loc = 'lower right')
    
    font_size_label = 13
    ax.set_xlabel('Mesh size requested (km)', fontsize = font_size_label)
    ax.set_ylabel('Mesh size produced (km)', fontsize = font_size_label)

    ax.set_aspect(1.0)

    ax.set_xlim(mesh_size_lims)

    plt.tight_layout()

    plt.savefig('gmsh_ellipsoid_surfaces.png', dpi = 300)

    plt.show()

    return

if __name__ == '__main__':

    main()

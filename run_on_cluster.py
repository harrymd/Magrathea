import os
from shutil import copyfile

import numpy as np

def mkdir_if_not_exist(dir_):

    if not os.path.exists(dir_):

        os.mkdir(dir_)

    return

def main():

    dir_work = '/work/06414/tg857131/Magrathea/'
    dir_cluster_input = os.path.join(dir_work, 'input_cluster')
    path_cluster_input = os.path.join(dir_cluster_input, 'input_cluster.txt')

    #dir_scratch = '/scratch/06414/tg857131/Magrathea/'
    dir_scratch = '/scratch/06414/tg857131/Magrathea/v6'

    file_model = 'prem_no_80km_03.0.txt'

    # Load cluster input file.
    with open(path_cluster_input, 'r') as in_id:

        tet_max_vol = float(in_id.readline().split()[1])
        order       = int(in_id.readline().split()[1])
        is_spherical= bool(int(in_id.readline().split()[1]))
        is_ellipsoidal = not(is_spherical)
        get_gravity = bool(int(in_id.readline().split()[1]))

    print('Read cluster input file {:}'.format(path_cluster_input))
    print('Maximum tetrahedron volume: {:>.3e} km3'.format(tet_max_vol))
    print('Finite-element order: {:>1d}'.format(order))
    print('Model is spherical (not oblate): {:}'.format(is_spherical))
    print('Calculate gravity: {:}'.format(get_gravity))

    if is_ellipsoidal:

        path_ellipticity = os.path.join(dir_cluster_input, 'ellipticity_profile.txt')
        ellipticity_data = np.loadtxt(path_ellipticity)
        r_ellipticity, ellipticity = ellipticity_data.T
        r_ellipticity = r_ellipticity*1.0E-3
        max_ellipticity = np.max(ellipticity)

    else:

        max_ellipticity = 0.0

    if max_ellipticity == 0.0:

        ellipticity_str = 'sph'

    else:

        ellipticity_str = '{:>3d}'.format(int(round(1.0/max_ellipticity)))

    mesh_size = 0.5*(2.0**(1.0/2.0))*(3.0**(1.0/3.0))*(tet_max_vol**(1.0/3.0))
    name_run = 'prem_{:>06.1f}_{:}'.format(mesh_size, ellipticity_str)

    dir_run = os.path.join(dir_scratch, name_run)
    dir_input = os.path.join(dir_run, 'input')
    path_input = os.path.join(dir_input, 'input.txt')
    #dir_output = os.path.join(dir_run, 'output')

    for dir_ in [dir_scratch, dir_run, dir_input]:#, dir_output]:

        mkdir_if_not_exist(dir_)

    with open(path_input, 'w') as out_id:

        out_id.write('dir_input {:}\n'.format(dir_input))
        out_id.write('dir_output {:}\n'.format(dir_scratch))
        out_id.write('file_model {:}\n'.format(file_model))
        out_id.write('tet_max_vol {:>16.12e}\n'.format(tet_max_vol))
        out_id.write('pOrder {:>1d}\n'.format(order))
        out_id.write('is_spherical {:>1d}\n'.format(int(is_spherical)))
        out_id.write('get_gravity {:>1d}\n'.format(int(get_gravity)))
    
    #name_model = 'prem_noocean.txt'
    #name_model = 'prem_no_80km_03.0.txt'
    name_outline = 'llsvp_smooth.txt'
    name_radii = 'radii.txt'
    name_ellipticity = 'ellipticity_profile.txt'

    names = [file_model, name_outline, name_radii, name_ellipticity]
    for name in names:

        path_source = os.path.join(dir_cluster_input, name)
        path_dest   = os.path.join(dir_input, name)

        copyfile(path_source, path_dest) 

    cmd_build_ellipsoidal = os.path.join(dir_work, 'build_planet.py')    
    cmd = 'python3 {:} {:}'.format(cmd_build_ellipsoidal, path_input)
    print(cmd)
    os.system(cmd)

    return

if __name__ == '__main__':

    main()

def main():

    # Load cluster input file.
    with open('input_cluster/input_cluster.txt', 'r') as in_id:

        tet_max_vol = float(in_id.readline())
        order       = 1

    dir_run_base = '/scratch/06414/tg857131/Magrathea/'

    mesh_size = 0.5*(2.0**(1.0/2.0))*(3.0**(1.0/3.0))*(tet_max_vol**(1.0/3.0))
    name_run = 'prem_{:>06.1f}_{1d}'.format(edge_length, order)

    dir_run = os.path.join(dir_run_base, name_run)

    mkdir_if_not_exist

    pass

if __name__ == '__main__':

    main()

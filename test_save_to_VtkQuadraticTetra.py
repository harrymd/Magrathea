import numpy as np 

# Some issues importing evtk modules, so do it like this:
import evtk
unstructuredGridToVTK = evtk.hl.unstructuredGridToVTK
VtkTriangle = evtk.vtk.VtkTriangle
VtkTetra = evtk.vtk.VtkTetra
VtkQuad = evtk.vtk.VtkQuad
VtkQuadraticTetra = evtk.vtk.VtkQuadraticTetra

def main():

    cell_id = VtkQuadraticTetra.tid

    n_tets = 2

    #pts = np.array([[0.0, 0.0, -1.0], # Point 1
    #                [0.0, 1.0,  0.0], # Point 2/6.
    #                [1.0, 0.0,  0.0], # Point 3/7.
    #                [0.0, 0.0,  0.0], # Point 4/8.
    #                [0.0, 0.0,  1.0], # Point 5.
    #                [0.0, 1.0,  0.0], # Point 2/6.
    #                [1.0, 0.0,  0.0], # Point 3/7.
    #                [0.0, 0.0,  0.0], # Point 4/8.
    #                ])

    pts = np.array([[2.0, 0.0, 0.0], #  1
                    [1.0, 1.0, 0.0], #  2
                    [0.0, 2.0, 0.0], #  3
                    [0.0, 1.0, 1.0], #  4
                    [0.0, 0.0, 2.0], #  5
                    [1.0, 0.0, 1.0], #  6
                    [1.0, 0.0, 0.0], #  7
                    [0.0, 0.0, 0.0], #  8
                    [0.0, 1.0, 0.0], #  9
                    [0.0, 0.0, 1.0], # 10
                    [2.0, 0.0, 0.0], #  1, 11
                    [1.0, 0.0, 0.0], #  7, 12
                    [0.0, 0.0, 0.0], #  8, 13
                    [0.0, 0.0, 1.0], # 10, 14
                    [0.0, 0.0, 2.0], #  5, 15
                    [0.0, -1.0, 1.0], # 16
                    [0.0, -2.0, 0.0], # 17
                    [0.0, -1.0, 1.0], # 18
                    [1.0, 0.0, 1.0], #  6, 19
                    [0.0, -1.0, 0.0],]) # 20


    x, y, z = pts.T

    cell_types = np.zeros(n_tets, dtype = np.int) + cell_id
    #connectivity = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype = np.int)
    connectivity = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype = np.int)
    np.random.shuffle(connectivity)
    offsets = np.array([10, 20])

    x       = x     .flatten(order = 'F')
    y       = y     .flatten(order = 'F')
    z       = z     .flatten(order = 'F')
    #tets    = tets  .flatten(order = 'F')
    #links   = links .flatten(order = 'F')
    connectivity = connectivity.flatten(order = 'F')
    #v_p     = v_p   .flatten(order = 'F')
    #v_s     = v_s   .flatten(order = 'F')
    #rho     = rho   .flatten(order = 'F')
    
    path_vtk = 'test'
    #unstructuredGridToVTK(
    #        path_vtk,
    #        x, y, z,
    #        connectivity    = links,
    #        offsets         = offsets,
    #        cell_types      = cell_types,
    #        cellData        = tet_data,
    #        pointData       = point_data)

    unstructuredGridToVTK(
            path_vtk,
            x, y, z,
            offsets         = offsets,
            cell_types      = cell_types,
            connectivity    = connectivity)

    return

if __name__ == '__main__':

    main()

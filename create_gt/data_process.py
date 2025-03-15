import numpy as np
import os
import open3d as o3d
import trimesh
import time

# all
def build_tet_adj_facet(input_path):
    for dir in os.listdir(input_path):
        file_path = input_path + "/" + dir + "/"
        file_names = dict()
        file_names['output_tetrahedron_adj'] = (os.path.join(file_path, "output_tetrahedron_adj"))
        file_names['output_facet_nei_cell'] = (os.path.join(file_path, "output_facet_nei_cell.txt"))

        tetrahedron_adj = np.fromfile(file_names['output_tetrahedron_adj'], dtype=np.int32, sep=' ').reshape(-1, 4)
        facet_nei_cell = np.fromfile(file_names['output_facet_nei_cell'], dtype=np.int32, sep=' ').reshape(-1, 2)
        facet_nei_cell = np.sort(facet_nei_cell, axis=1)
        facet_idx = dict()
        for i in range(facet_nei_cell.shape[0]):
            string = str(facet_nei_cell[i][0]) + "--" + str(facet_nei_cell[i][1])
            facet_idx[string] = i

        tet_idx_key = np.linspace(0, tetrahedron_adj.shape[0]-1, tetrahedron_adj.shape[0]).astype(np.int32).reshape(-1,1)
        tet_neighbor = dict()
        tet_neighbor[0] = np.sort(np.hstack((tet_idx_key, tetrahedron_adj[:,0].reshape(-1,1))), axis=1)
        tet_neighbor[1] = np.sort(np.hstack((tet_idx_key, tetrahedron_adj[:,1].reshape(-1,1))), axis=1)
        tet_neighbor[2] = np.sort(np.hstack((tet_idx_key, tetrahedron_adj[:,2].reshape(-1,1))), axis=1)
        tet_neighbor[3] = np.sort(np.hstack((tet_idx_key, tetrahedron_adj[:,3].reshape(-1,1))), axis=1)

        tet_adj_facet = (np.ones((tetrahedron_adj.shape[0], 4)) * (-1)).astype(np.int32)
        for j in range(4):
            tet = tet_neighbor[j]
            for i in range(tet.shape[0]):
                string = str(tet[i][0]) + "--" + str(tet[i][1])
                idx = facet_idx.get(string)
                if idx is None:
                    tet_adj_facet[i][j] = -1
                else:
                    tet_adj_facet[i][j] = idx
        np.savetxt(os.path.join(file_path, 'tet_adj_facet.txt'), tet_adj_facet, fmt='%d')

        file_names.clear()
        facet_idx.clear()
        tet_neighbor.clear()
        print(dir, "build_tet_adj_facet done")


if __name__ == "__main__":
    input_path = "../example/processed_data/"
    build_tet_adj_facet(input_path)


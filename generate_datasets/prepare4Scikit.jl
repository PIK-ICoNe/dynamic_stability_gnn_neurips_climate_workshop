using Pkg
# Pkg.activate("../")
Pkg.activate(".")

using HDF5
using LightGraphs, EmbeddedGraphs
using StatsBase

runs_grids = 10000
const grid_index_start = 1 
const grid_index_end = 10000
const K = 9.

grid_path = "/home/nauck/projects/dynamic_stability_dataset/snbs_homogeneous_dataset/SN_N020_pert10000/grids/grids/"
scikit_path = "/home/nauck/projects/dynamic_stability_dataset/snbs_homogeneous_dataset/SN_N020_pert10000/scikit_data/"

# regenerate_graph a graph of type EmbeddedGraphs
function regenerate_graph(vertices, grids_weights_col_ptr, grids_weights_row_val)
    new_graph = EmbeddedGraph(SimpleGraph(0), [])
    for i in 1:size(vertices, 1)
        add_vertex!(new_graph, vertices[i,:])
    end
    for i in 1:length(grids_weights_col_ptr)
        EmbeddedGraphs.weights(new_graph).colptr[i] = grids_weights_col_ptr[i]
    end
    for i in 1:length(grids_weights_col_ptr) - 1
        for j in
            grids_weights_col_ptr[i]:grids_weights_col_ptr[i + 1] - 1
            add_edge!(new_graph, grids_weights_row_val[j], i)
        end
    end
    return new_graph
end


# generate input data for GNN: grids and P
for r in grid_index_start:grid_index_end
    id = lpad(r, length(digits(runs_grids)), '0')
    # file_name_read = string("grids/grids/grid_", id, ".h5")
    file_name_read = string(grid_path,string("grid_", id, ".h5"))
    grids_vertices, grids_weights_col_ptr, grids_weights_row_val, P = h5open(file_name_read, "r") do file
        read(file, "grids_vertices", "grids_weights_col_ptr", "grids_weights_row_val", "P")
    end
    g = regenerate_graph(grids_vertices, grids_weights_col_ptr, grids_weights_row_val)
    a = K * adjacency_matrix(g)
    A =  Array(a)
    # file_name_scikit = string("file_name_scikit/grid_data_", id, ".h5")
    file_name_scikit = string(scikit_path,string("grid_scikit_", id, ".h5"))
    h5open(file_name_scikit, "w") do file
        write(file, "adjacency_matrix", A)
	write(file, "P", P)
    end
end

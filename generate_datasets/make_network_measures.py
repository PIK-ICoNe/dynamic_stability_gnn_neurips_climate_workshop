import h5py
import numpy as np
import pandas as pd


# networkx
import networkx as nx

# from node_classification import node_categories
# plot
import matplotlib.pyplot as plt

ds20_scikit_path = '../../datasets/dataset020/scikit_data/'
ds100_scikit_path = '../../datasets/dataset100/scikit_data/'
texas_scikit_path = '../../datasets/dataset_texas/scikit_data/'

num_grids_max_index = 10000
num_grids_texas = 1

ds_format_number_grids_input = '5'
texas_format_number_grids_input= '1'

list_measures = ["P",
    "degree",
    "average_neighbor_degree",
    "clustering",
    "current_flow_betweenness_centrality",
    "closeness_centrality"
    ]

# list_name_node_categories = ['bulk', 'root', 'dense sprout', 'sparse sprout', 'inner tree node', 'proper leaf']
# add_node_cat = True

class network_measures():
    def __init__(self):
        super(network_measures, self).__init__()
        self.network_measures = {}
        # self.list_node_categories = list_name_node_categories
    
    def convert_degree_view_to_list(self,degreeview):
        new_list = []
        for i in range(len(degreeview)):
            new_list.append(degreeview[i])
        return new_list      
    
    def compute_measure(self,G,A,P,measure):
        if measure == 'P':
            self.network_measures["P"] = P
        elif measure == "degree":
            dict_degree = nx.degree(G)
            degree_list = self.convert_degree_view_to_list(dict_degree)
            self.network_measures["degree"] = degree_list       
        elif measure == "clustering":
            self.network_measures["clustering"] = list(nx.clustering(G).values()) 
        elif measure == "closeness_centrality":
            self.network_measures["closeness_centrality"] = list(nx.closeness_centrality(G).values())
        elif measure == "current_flow_betweenness_centrality":
            self.network_measures["current_flow_betweenness_centrality"] = list(nx.current_flow_betweenness_centrality(G).values())
        elif measure == "average_neighbor_degree":
            self.network_measures["average_neighbor_degree"] = list(nx.average_neighbor_degree(G).values())
        
        
    def compute_list_measures(self,G,A, P, list_measures):
        for i in range(len(list_measures)):
            self.compute_measure(G,A,P,list_measures[i])
        # if node_cat:
        #     self.add_node_cats(G)
    
    def return_dataframe_measures(self,num_nodes):
        network_measure_names = list(self.network_measures.keys())
        dataframe = pd.DataFrame(columns=network_measure_names, index=np.arange(1,num_nodes+1))
        dataframe.index.name = "node"
        num_measures = len(network_measure_names)
        for i in range(num_nodes):
            for j in range(num_measures):
                measure = network_measure_names[j]
                dataframe.at[i+1,measure] = self.network_measures[measure][i]
        return dataframe
    
    def get_min_max_measure(self,measure):
        min_value = min(self.network_measures[measure])
        max_value = max(self.network_measures[measure])
        return min_value, max_value
         
    def get_measure_bounds(self):
        network_measure_names = list(self.network_measures.keys())
        num_measures = len(network_measure_names)
        dataframe = pd.DataFrame(columns=network_measure_names, index=["min", "max"])
        for measure in self.network_measures:
            min_value, max_value = self.get_min_max_measure(measure)
            dataframe.at["min", measure] = min_value
            dataframe.at["max", measure] = max_value
        return dataframe
    
    def compute_neighbor_degree_min_max(self,G):
        degree_neighbors = []
        dict_degree = nx.degree(G)
        degree_list = self.convert_degree_view_to_list(dict_degree)
        min_degree_neighbor = []
        max_degree_neighbor = []
        for i in range(len(G.nodes)):
            degree_neighbors = []
            for n in G.neighbors(i):
                degree_neighbors.append(degree_list[n]) 
            min_degree_neighbor.append(min(degree_neighbors))
            max_degree_neighbor.append(max(degree_neighbors))
        return min_degree_neighbor, max_degree_neighbor
        
def generate_graph(index,path_dir, format_number_grids):
    id = format(index, '0'+format_number_grids)
    file_to_read = path_dir + 'grid_scikit_'+str(id) + '.h5'
    hf = h5py.File(file_to_read, 'r')
    A = hf.get('adjacency_matrix')
    A = np.array(A)
    P = hf.get('P')
    P = np.array(P)
    hf.close()
    num_nodes = len(A)
    G = nx.convert_matrix.from_numpy_matrix(A, parallel_edges=False, create_using=None)
    # add node properties
    P_dict = dict(enumerate(P.flatten(), 0))
    nx.set_node_attributes(G, P_dict, "P")
    return G, A, P
    
        
def update_bounds(computed_boundaries, previous_boundaries):
    names = list(computed_boundaries.columns)
    for measure in names:
        bds_prev = previous_boundaries[measure]
        bds_computed = computed_boundaries[measure]
        if bds_prev["min"] < bds_computed["min"]:
            bds_computed["min"] = bds_prev["min"]
        if bds_prev["max"] > bds_computed["max"]:
            bds_computed["max"] = bds_prev["max"]
    return computed_boundaries 
    
def get_boundaries_of_measures(index_start, index_end, list_measures, input_path_dir, format_number_grids):
    G1, A1, P1 = generate_graph(index_start, input_path_dir, format_number_grids)
    grid_measures = network_measures()
    grid_measures.compute_list_measures(G1,A1, P1, list_measures)
    computed_bounds = grid_measures.get_measure_bounds()
    for i in range(index_start+1,index_end+1):
        G, A, P = generate_graph(i, input_path_dir, format_number_grids)
        grid_measures = network_measures()
        grid_measures.compute_list_measures(G,A,P,list_measures)
        boundaries = grid_measures.get_measure_bounds()
        computed_boundaries = update_bounds(computed_bounds,boundaries)
    return computed_bounds

def normalize_measures(measures,boundaries):
    for measure in list(measures.columns):
        array= measures[measure]
        # not_normalizable = list_name_node_categories.copy()
        # not_normalizable.append("node_cat")
        # if measure not in not_normalizable: 
        bounds = boundaries[measure]
        for i in array.index:
            array[i] = (array[i] - bounds["min"])/(bounds["max"]-bounds["min"])
    return measures
        

def export_measures(index_start, index_end, input_path_dir, format_number_grids, boundaries=False):    
    path_output_dir = input_path_dir
    for i in range(index_start,index_end+1):
        G, A, P = generate_graph(i, input_path_dir, format_number_grids)
        grid_measures = network_measures()
        grid_measures.compute_list_measures(G,A,P,list_measures)
        num_nodes = G.number_of_nodes()
        dataframe_measures = grid_measures.return_dataframe_measures(num_nodes)
        id = format(i, '0'+format_number_grids)
        if type(boundaries) == bool:
            dataframe_measures.to_csv(path_output_dir + 'network_measures_not_normalized'+str(id) + '.csv')
        else:
            normalized_measures = normalize_measures(dataframe_measures,boundaries)
            normalized_measures.to_csv(path_output_dir + 'network_measures_'+str(id) + '.csv')
            

print("init finished")

computed_bounds_20 = get_boundaries_of_measures(1, num_grids_max_index, list_measures, ds20_scikit_path, ds_format_number_grids_input)
print("computation of bounds20 finished")
computed_bounds_100 = get_boundaries_of_measures(1, num_grids_max_index, list_measures, ds100_scikit_path, ds_format_number_grids_input)
print("computation of bounds1000 finished")
computed_bounds_texas = get_boundaries_of_measures(1, num_grids_texas, list_measures, texas_scikit_path, texas_format_number_grids_input)

print("computation of bounds finished")
computed_bounds = update_bounds(computed_bounds_20, computed_bounds_100)
computed_bounds_final = update_bounds(computed_bounds_texas, computed_bounds)


#export_measures(1,num_grids_max_index, ds20_scikit_path, ds_format_number_grids_input, boundaries=computed_bounds_final)
#export_measures(1,num_grids_max_index, ds100_scikit_path, ds_format_number_grids_input, boundaries=computed_bounds_final)
export_measures(1,1, texas_scikit_path, texas_format_number_grids_input, boundaries=computed_bounds_final)

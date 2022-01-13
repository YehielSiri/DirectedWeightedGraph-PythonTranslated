# DirectedWeightedGraph.py

# introduction

This repository implements the directed weighted graph in Python as the third assignment of Object Oriented Programming curse 
in Ariel University. In fact, this is the second assignment translated (from Java) to Python.
This project purpose is to compare between a Java code version to its Python translate. The Python version is this repository 
and the Java one is in link-
https://github.com/YehielSiri/DirectedWeightedGraph

# Installation

Firstlly, to clone the project into your local workspace, copy the command below to the Bash git on your directory:
<git clone https://github.com/YehielSiri/DirectedWeightedGraph-PythonTranslated.git>
Then, execute the main.py file.

# How to run

In the first step, a directed weighted graph must be created - For example:

g = DiGraph() graph.add_node(id key=0)-->key number 0 graph.add_node(id key=1)-->key number 1 graph.add_node(id key=2)-->key number 2 graph.add_node(id key=3)-->key number 3

graph.add_edge(0, 1, 1)
graph.add_edge(1, 0, 1.1)
graph.add_edge(1, 2, 1.3)
graph.add_edge(2, 3, 1.1)
graph.add_edge(1, 3, 10)

Now, you can use all the functions of the class DiGraph. For example:

graph.get_all_v()  output-->     {0:0 , 1: 1: ,2: 2: ,3: 3}
graph.all_in_edges_of_node(1) output--> {0: 1}
graph.all_out_edges_of_node(1) output--> {0: 1.1, 2: 1.3, 3: 10}

At this point we creates a Graph_Algo object:

algo = Graph_Algo(graph).

Now, you can use all the functions algoritem of the class. For example:

algo.shortest_path(0, 3) output-->  (3.4, [0, 1, 2, 3])
algo.connected_component(id=1) utput--> [0, 1]
algo.connected_components()  output-->[[3], [2], [1, 0]]
algo.save_to_json("../file//testGraph1").
algo.load_from_json("../file//testGraph1").
algo.plot_graph() output-->

# Data structures and classes details

# Node class:

This object represents a vertex in a directed weighted graph.

__init__(node, pos) - init the node; its position, its neighbors in & out etc.
__iter__() - return an iterator to the neighbors.
__repr__() - return a string representation of the node by __str__() (an auxiliary function).
__str__() - return node as string.
get_id() - return this node's key.
get_weight(neighbor) - return the weight on the edge is which out of this node to the neighbor which is given.
getNi() - return all the neighbors into this node.
get_weightIn(neighbor) - return the weight on the edge is which into this node from the neighbor which is given.
deleteNi_in(id1) - delete "an into neighbor".
deleteNi_out(id1) - delete "an out from neighbor".
add_out_Ni(node_out, weight) - add "an out from neighbor" to the current node with a weight.
add_in_Ni(node_in, weight) - add "an into neighbor" to the current node with a weight.


# DiGraph:

A directed weighted graph representation class-object by implementing DirectedWeightedGraph interface. This interface should support a large number of nodes, and hence should be based on an efficient compact representation (not on a n*n one). So the implement based on HushMap (time complexity - O(1) & space complexity - dynamically).

__init__() - init the graph.
__iter__() - return an iterator.
get_vertex(n) - get a Node which exists, else, return NONE.
v_size() -> int - returns number of vertices.
e_size() -> int - returns number of edges.
get_all_v() -> dict - return all the nodes as a dictionary, (key, node_data) for each node.
all_in_edges_of_node(id1) -> dict - return all the nodes which are connected into 'node_id', (key, weight) for each node.
all_out_edges_of_node(id1) -> dict - return all the nodes which are connected from 'node_id', (key, weight) for each node.
add_edge(id1, id2, weight) -> bool - addind an edge: start node, end node, weight and return True if the edge was added successfully.
                                     Note: If the edge already exists or one of the nodes dose not exists the functions will do nothing
add_node(node_id, pos) -> bool - adding a node: node ID, position and return True if the node was added successfully.
remove_node(node_id) -> bool - removes a node: node ID and return True if the node was removed successfully.
                               Note: If the node does not exists the function will do nothing.
remove_edge(node_id1, node_id2) -> bool - removes an edge: start node, end node and return True if the edge was removed successfully.
                               Note: If the edge does not exists the function will do nothing.


# GraphAlgo:

Class for solving an algorithmic problems as, Shortest path, Is a connective graph, What is the center, TSP (include read graph from a json file and save to).

__init__(algo) - init the graph.
get_graph() - return the directed weighted graph on which the algorithm is working on.
load_from_json(file_name) - loads a graph from a json file by path to the file and return True if it has secceeded.
save_to_json(file_name) - saves a graph to a json file by path to the file and return True if it has secceeded.
shortest_path(id1, id2) -> (float, list) - finding the shortest path from a node to another using Dijkstra algorithm.
TSP(nodes list) -> (nodes path list, float) - finding the shortest path which visits all the nodes.
centerPoint() -> (center ID, float) - finding the node that has the shortest distance to it's farthest node.
plot_graph() -> None - ploting the graph. If the nodes have a position, the nodes will be placed there. Otherwise, they will be placed in a random but elegant manner.



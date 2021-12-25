from abc import ABC
from collections import defaultdict

from Node import Node

from GraphInteface import GraphInterface



class DirectedWeightedGraph(ABC, GraphInterface):

	def __init__(self):
		self.DirectedWeightedGraph = dict()
		self.num_vertices = 0
		self.Mc = 0
		self.edge_size = 0

	def __iter__(self):
		return iter(self.DirectedWeightedGraph.values())

	def get_vertex(self, n):
		if n in self.DirectedWeightedGraph:
			return self.DirectedWeightedGraph[n]
		else:
			return None
	"""
	get a node if the node exist in the graph, else, return None
	"""


	def v_size(self) -> int:
		return self.num_vertices
	"""
	Returns the number of vertices in this graph
	@return: The number of vertices in this graph
	"""

	def e_size(self) -> int:
		return self.edge_size
	"""
	Returns the number of edges in this graph
	@return: The number of edges in this graph
	"""

	def get_all_v(self) -> dict:
		return self.get_vertices()

	"""
	return a dictionary of all the nodes in the Graph, each node is represented using a pair (node_id, node_data)
	"""

	def all_in_edges_of_node(self, id1: int) -> dict:
		if self.get_vertex(id1) is not None:
			return self.get_vertex(id1).Ni_node_in

	"""
	return a dictionary of all the nodes connected to (into) node_id ,
	each node is represented using a pair (other_node_id, weight)
	"""

	def all_out_edges_of_node(self, id1: int) -> dict:
		if self.get_vertex(id1) is not None:
			return self.get_vertex(id1).Ni_node_out

	"""
	return a dictionary of all the nodes connected from node_id , each node is represented using a pair
	(other_node_id, weight)
	"""

	def get_mc(self) -> int:
		return self.Mc
	"""
	Returns the current version of this graph,
	on every change in the graph state - the MC should be increased
	@return: The current version of this graph.
	"""


	def add_edge(self, id1: int, id2: int, weight: float) -> bool:
		if id2 in self.get_vertex(id1).Ni_node_out:
			return False
		if weight <= 0 or id1 == id2:
			return False
		boolAdd = False

		if id1 not in self.DirectedWeightedGraph:
			return boolAdd
		if id2 not in self.DirectedWeightedGraph:
			return boolAdd
		
		boolAdd = True

		self.DirectedWeightedGraph[id1].add_out_Ni(id2, weight)
		self.DirectedWeightedGraph[id2].add_in_Ni(id1, weight)

		self.edge_size = self.edge_size + 1
		self.Mc += 1
		return boolAdd

	"""
	Adds an edge to the graph.
	@param id1: The start node of the edge
	@param id2: The end node of the edge
	@param weight: The weight of the edge
	@return: True if the edge was added successfully, False o.w.
	Note: If the edge already exists or one of the nodes dose not exists the functions will do nothing
	"""


	def add_node(self, node_id: int, pos: tuple = None) -> bool:
		x = False
		if node_id in self.DirectedWeightedGraph:
			return x
		x = True
		self.num_vertices = self.num_vertices + 1
		new_Node = Node(node_id, pos)
		self.Mc = self.Mc + 1
		self.DirectedWeightedGraph[node_id] = new_Node

		return x

	"""
	Adds a node to the graph.
	@param node_id: The node ID
	@param pos: The position of the node
	@return: True if the node was added successfully, False o.w.
	Note: if the node id already exists the node will not be added
	"""

    def remove_node(self, node_id: int) -> bool:
		bool1 = False
		if node_id not in self.DirectedWeightedGraph:
			return bool1

		if len(self.get_vertex(node_id).Ni_node_in )> 0:
			for node in self.DirectedWeightedGraph[node_id].Ni_node_in.keys():
				self.DirectedWeightedGraph[node].Ni_node_out.pop(node_id)
				self.edge_size -= 1

		if len(self.get_vertex(node_id).Ni_node_out) > 0:
			for node in self.DirectedWeightedGraph[node_id].Ni_node_out.keys():
				self.DirectedWeightedGraph[node].Ni_node_in.pop(node_id)
				self.edge_size -= 1

		self.Mc += 1
		self.num_vertices = self.num_vertices - 1
		self.DirectedWeightedGraph.pop(node_id)
		return bool1

	"""
	Removes a node from the graph.
	@param node_id: The node ID
	@return: True if the node was removed successfully, False o.w.
	Note: if the node id does not exists the function will do nothing
	"""


	def remove_edge(self, node_id1: int, node_id2: int) -> bool:
		boolRemove = False
		if self.get_vertex(node_id1) is None or self.get_vertex(node_id2) is None or node_id1 == node_id2:
			return boolRemove

		if node_id1 not in self.DirectedWeightedGraph:
			return boolRemove

		if node_id2 not in self.DirectedWeightedGraph:
			return boolRemove
		if node_id1 not in self.get_vertex(node_id2).Ni_node_in or node_id2 not in self.get_vertex(node_id1).Ni_node_out:
			return boolRemove
		self.get_vertex(node_id2).deleteNi_in(node_id1)
		self.get_vertex(node_id1).deleteNi_out(node_id2)

		boolRemove = True

		self.edge_size -= 1
		self.Mc += 1
		return boolRemove

	"""
	Removes an edge from the graph.
	@param node_id1: The start node of the edge
	@param node_id2: The end node of the edge
	@return: True if the edge was removed successfully, False o.w.
	Note: If such an edge does not exists the function will do nothing
	"""

"""
Auxiliary functions:
"""

	def __str__(self):
		out_str = 'DirectedWeightedGraph:(|V|=' + str(self.v_size()) + ',' + '|E|=' + str(self.e_size()) + ')'
		for node in self.DirectedWeightedGraph:
			out_str += '{' + 'Node :' + 'key-> ' + node.__str__() + '},'
		return out_str

	def __contains__(self, key):
		return key in self.DirectedWeightedGraph.keys()

	def __iter__(self):
		return iter(self.DirectedWeightedGraph.values())

	def get_vertex_list(self) -> list:
		list1 = []
		dict1 = self.get_all_v()
		for i in dict1:
			list1.append(dict1.get(i))
		return list1

	def get_vertices(self):
		ordered_vertices = {}
		for vertex in self.DirectedWeightedGraph:
			ordered_vertices[vertex] = str(self.DirectedWeightedGraph[vertex].id) + '->' + ' |edges out|: ' + str(
				self.DirectedWeightedGraph[vertex].Ni_node_out) + ',' + str(self.DirectedWeightedGraph[vertex].id) + '<-' + '|edges in|: ' + str(
				self.DirectedWeightedGraph[vertex].Ni_node_in)

		return ordered_vertices
	"""
	Returns a dict of nodes representing the vertices in the graph, in dict order
	"""

	def __repr__(self):
		return self.__str__()
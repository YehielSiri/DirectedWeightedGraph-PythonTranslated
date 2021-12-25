import random


class NodeData:
	"""
	init the Node
	"""
	def __init__(self, node, pos: tuple = None):
		self,id = node

		self.Ni_node_out = dict()
		self.Ni_node_in = dict()
		self.position = pos

		if pos is None:
			x = 35.185+(0.03)*random.random()
			y = 35.185+(0.03)*random.random()
			newPos = (str(x), str(y))
			self.position = newPos
		self.visited = False
		self.color = "white"  # 0 (white) means not visited yet, 1 (grey) means visited but didn't visit all neighbors,
		# 2 (black) means visited and visited all neighbors
		self.dist = None  # used in the minimum path algorithms.
		self.info = ""
		self.toWeight = 0.0
		self.Tag = 0

	def __iter__(self):
		return iter(self.Ni.keys())

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		retests = '[' + str(self.id) + ']' + ' -> ( '
		for i in self.Ni_node_out:
			retests = retests + str(i) + ' : ' + 'W' + str([self.get_weight(i)]) + ', '
		retests += ')'
		for k in self.Ni_node_in:
			retests = retests + ' , ' + '(' + str(k) + ' '
			retests = retests + ')  -> ' + '[' + str(self.id) + ' : ' + 'W' + str([self.get_weightIn(k)]) + ', ' + ']'
		return retests
	"""""
	return the id of this Node
	"""

	def get_id(self):
		return self.id

	"""
	by given int neighbor get the weight in NIout of this node and this int neighbor
	"""

	def get_weight(self, neighbor):
		if neighbor in self.Ni_node_out:
			return self.Ni_node_out[neighbor]
		return None



	def getNi(self):
		return self.Ni

	"""
	by given int neighbor get the weight in NIin of this node and this int neighbor
	"""

	def get_weightIn(self, neighbor):
		if neighbor in self.Ni_node_in:
			return self.Ni_node_in[neighbor]
		return None

	"""
	by given int id1 delete him from neigbors in to this node
	"""

	def deleteNi_in(self, id1):
		del self.Ni_node_in[id1]
	"""
	by given int id1 delete him from neigbors out to this node
	"""

	def deleteNi_out(self, id1):
		self.Ni_node_out.pop(id1)

	def getVisited(self):
		return self.visited

	def setVisited(self, val=True):
		self.visited = val

	def setPosition(self, x, y, z):
		self.position = (x, y, z)

	def getPosition(self):
		return self.position

	"""
	by given int id1 and float weight add to neighbor out of this node a ni and weight between them
	"""

	def add_out_Ni(self, node_out: int, weight: float):
		self.Ni_node_out[node_out] = weight
	"""
	by given int id1 and float weight add to neighbor in of this node a ni and weight between them
	"""

	def add_in_Ni(self, node_in: int, weight: float):
		self.Ni_node_in[node_in] = weight

	""""
	def __eq__(self, other):
		return self.id == other.id and self.position == other.position \
			and self.node_in == other.node_in and self.node_out == other.node_out
	"""
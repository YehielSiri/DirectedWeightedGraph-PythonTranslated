from typing import List
from matplotlib.widgets import Cursor
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.style as plt1
import numpy as np

from GraphAlgoInterface import GraphAlgoInterface

from Node import Node
from DiGraph import DiGraph

try:
    import json
except ImportError:
    import simplejson as json


class GraphAlgo1():
    graph = dict()
    graph['Nodes'] = list()
    graph['Edges'] = list()


def diagram_compare():
    with plt.style.context('ggplot'):
        names = ['java', 'python', 'networkx']
        colors = mcolors.BASE_COLORS
        times1 = [0.0008211099700927734, 0.0007131099700927734, 0.00012493133544921875]
        times2 = [0.000156, 0.000114, 4.029273986816406e-05]
        times3 = [0.01654395481372, 0.01556396484375, 1.0967254638671875e-05]
        # plt.plot(9 , 9 , 9 , 9)
        plt.figure(figsize=(15, 7))
        plt.subplot(131, title='Connected component(1)')
        plt.bar(names, times1, color=colors)
        plt.subplot(132, title='Shortest_path(2,5)')
        plt.bar(names, times2, color=colors)
        plt.subplot(133, title='Connected components')
        plt.bar(names, times3, color=colors)
        plt.suptitle('Running time Tests - DirectedWeightedGraph(File Graph ; data/G_10_80_0.json)\nRun on Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz   2.11 GHz processor, 7.86 GB usable installed RAM\n')
        plt.show()


class GraphAlgo(GraphAlgoInterface):
    """
       init the graphAlgo
       """

    def __init__(self, algo: DiGraph = None):
        self.graphAlgo = algo
        self.counterComp = 0

        """
              :return: the directed graph on which the algorithm works on.
              """

    def get_graph(self):
        return self.graphAlgo

    """
              Loads a graph from a json file.
              @param file_name: The path to the json file
              @returns True if the loading was successful, False o.w.
              """

    def load_from_json(self, file_name: str):
        try:
            self.open_graph(file_name)
        except Exception as e:
            print(e)
            return False

        return True

    """
          Saves the graph in JSON format to a file
          @param file_name: The path to the out file
          @return: True if the save was successful, False o.w.
          """

    def save_to_json(self, file_name: str):
        graph = dict()
        graph['Nodes'] = list()
        graph['Edges'] = list()

        for node in self.graphAlgo.get_vertices():
            if self.graphAlgo.get_vertex(node).position is None:
                pos = '0.0,0.0,0.0'
            else:
                pos = str(str(self.graphAlgo.get_vertex(node).position[0]) + ',' + str(
                    self.graphAlgo.get_vertex(node).position[1]) + ',0.0')
            graph['Nodes'].append({"id": node, "pos": pos})
            elements = self.graphAlgo.get_vertex(node).Ni_node_out
            for child in elements:
                graph['Edges'].append(
                    {"src": node, "w": self.graphAlgo.get_vertex(node).get_weight(child), "dest": child})
        try:
            with open(file_name, 'w') as json_file:
                json.dump(graph, json_file)
                return True
        except IOError:
            return False

    """
         Returns the shortest path from node id1 to node id2 using Dijkstra's Algorithm
         @param id1: The start node id
         @param id2: The end node id
         @return: The distance of the path, the path as a list
         Example:
 #      >>> from GraphAlgo import GraphAlgo
 #       >>> g_algo = GraphAlgo()
 #        >>> g_algo.addNode(0)
 #        >>> g_algo.addNode(1)
 #        >>> g_algo.addNode(2)
 #        >>> g_algo.addEdge(0,1,1)
 #        >>> g_algo.addEdge(1,2,4)
 #        >>> g_algo.shortestPath(0,1)
 #        (1, [0, 1])
 #        >>> g_algo.shortestPath(0,2)
 #        (5, [0, 1, 2])
         More info:
         https://en.wikipedia.org/wiki/Dijkstra's_algorithm
         """

    def shortest_path(self, id1: int, id2: int) -> (float, list):
        # check for null
        val = self.shortestPath_Dist(id1, id2)
        if val == -1:
            return float('inf'), []
        # self.Dijkstra(id1, id2)

        # check val for -1 if not successfully arrived.
        curV = self.graphAlgo.get_vertex(id2)
        path = []
        path.insert(0, id2)
        while curV.id != id1:
            toAdd = int(str(curV.info))
            if toAdd not in path:
                path.append(toAdd)
            curV = self.graphAlgo.get_vertex(toAdd)
        path.reverse()

        return val, path


        """
        Finds the shortest path that visits all the nodes in the list
        :param node_lst: A list of nodes id's
        :return: A list of the nodes id's in the path, and the overall distance
        """
    def TSP(self, node_lst: List[int]) -> (List[int], float):
        start = time.time()
        solution = self.tsp(node_lst)
        if solution is None:
            return None, -1
        solutionCopy = copy.deepcopy(solution)
        cost = self.pathWeight(solutionCopy)
        # print("TSP has calculated in ", time.time() - start, "seconds with weight:", cost)
        return solution, cost
        # return self.fixPath(node_lst, solution), cost


        """
        Finds the node that has the shortest distance to it's farthest node.
        :return: The nodes id, min-maximum distance. If there is no center (an unconnected graph), return None,float('inf').
        """
    def centerPoint(self) -> (int, float):
        if self.is_connected():
            min1 = sys.float_info.max
            node_id = -1
            for k in self.graphAlgo.get_all_v():
                curr_node = k
                max1 = sys.float_info.min
                for v in self.graphAlgo.get_all_v():
                    if v == k:  # same node, no need to check again.
                        continue
                    next_node = v
                    tmp_dijk = self.shortest_path(curr_node, next_node)
                    tmp = tmp_dijk[1]
                    if tmp_dijk[0] is not INF:
                        if tmp > max1:
                            max1 = tmp
                        if tmp > min1:
                            # we want Minimum of all maximums
                            break

                if min1 > max1:
                    min1 = max1
                    node_id = k

            return node_id, min1
        else:
            return False









    """Auxiliary functions"""
    def open_graph(self, file_name):
        Mygraph = DiGraph()

        try:
            with open(file_name, "r") as json_file:
                json_graph = json.load(json_file)
                for i in json_graph['Nodes']:
                    if 'pos' in i:
                        strP = i["pos"].split(",")
                        x = float(strP[0])
                        y = float(strP[1])
                        Mygraph .add_node(i["id"], (x, y))
                    else:
                        Mygraph .add_node(i["id"])
                for j in json_graph["Edges"]:
                    Mygraph .add_edge(j["src"], j["dest"], j["w"])
        except IOError as e:
            print(e)
            return False
        self.graphAlgo = Mygraph
        return True

    """
     * returns the the shortest path between src to dest - as an ordered List of nodes:
     * src--> n1-->n2-->...dest
     * see: https://en.wikipedia.org/wiki/Shortest_path_problem
               @return: None
               """

    def shortestPath_Dist(self, id1: int, id2: int):
        if self.graphAlgo.get_vertex(id1) is None or self.graphAlgo.get_vertex(id2) is None:
            print("One or two of the inputs do not exist")
            return -1
        self.Dijkstra(id1, id2)
        if self.graphAlgo.get_vertex(id2).toWeight == float("inf"):
            print("There is not a path between the nodes : ")
            return -1
        return self.graphAlgo.get_vertex(id2).toWeight

    """
               Dijkstra Algorithem
               @return: None
               """

    def Dijkstra(self, source, destination):
        # x = self.graphAlgo.get_all_v()
        # v1 = Node(0)
        for v1 in self.graphAlgo.DirectedWeightedGraph:
            self.graphAlgo.get_vertex(v1).Tag = 0
            self.graphAlgo.get_vertex(v1).info = ""
            self.graphAlgo.get_vertex(v1).toWeight = float("inf")
        self.graphAlgo.get_vertex(source).toWeight = 0
        min_node = self.graphAlgo.get_vertex(source)
        prev_node = self.graphAlgo.get_vertex(source)
        while prev_node.id != destination and min_node.info != "empty":
            min_node.Tag = 1  # visited true.
            if len(min_node.Ni_node_out) > 0:  # KOBI
                for currE in min_node.Ni_node_out:
                    edgeWeight = self.graphAlgo.get_vertex(min_node.id).get_weight(currE)
                    if self.graphAlgo.get_vertex(
                            currE).Tag == 0 and min_node.toWeight + edgeWeight < self.graphAlgo.get_vertex(
                        currE).toWeight:
                        self.graphAlgo.get_vertex(currE).toWeight = min_node.toWeight + edgeWeight
                        self.graphAlgo.get_vertex(currE).info = str(min_node.id)  # from where:
                        prev_node = min_node

            min_node = self.find_Min_Node(self.graphAlgo.DirectedWeightedGraph)  # KOBI

            """
                       find Min Node
                       @return: node_return
                       """

    def find_Min_Node(self, vertex: dict):  # make sure vertex is node data
        node_return = Node(0)
        node_return.toWeight = float("inf")
        node_return.info = "empty"
        node_return.Tag = 1
        for curr in vertex:

            if self.graphAlgo.get_vertex(curr).Tag == 0 and self.graphAlgo.get_vertex(
                    curr).toWeight < node_return.toWeight:
                node_return = self.graphAlgo.get_vertex(curr)

        return node_return

    """
        TSP auxiliary functions
    """

    def pathWeight(self, path: list):
        path.reverse()
        ans = 0
        prevV = path.pop()
        while path:
            nextV = path.pop()
            ans += self.cost(prevV, nextV)
            prevV = nextV
        return ans

    def cost(self, node, neighbor):
        if node == neighbor:
            return 0.0
        neighbors = self.graphAlgo.all_out_edges_of_node(node)
        if len(neighbors) > 0:
            return neighbors.get(neighbor)
        return sys.float_info.max

        """
        Compute heuristically a path for TSP problem (variation of: can repeat visits)
        :param cities: list of nodes to visit
        :return: an ordered path
        """
    def tsp(self, cities: list) -> List:
        currentCity = cities.pop()
        path = [currentCity]

        while len(cities) > 0:

            if len(cities) == 1:
                last_city = cities.pop()
                temp = self.shortest_path(path[-1], last_city)
                if temp[0] == INF:
                    return None
                path_to_rel = tuple(temp)
                path_to_rel = path_to_rel[0]
                path_to_rel.reverse()
                path_to_rel.pop()
                while path_to_rel:
                    t = path_to_rel.pop()
                    path.append(t)
                    if t in cities:
                        cities.remove(t)
                    if len(cities) == 0:
                        break
                # to_visit.remove(min(to_visit))
                continue

            try:
                currentCity = self.getCheapestNeighbour(path[-1])  # arr[-1] accesses last member of array
            except:
                currentCity = -1

            if currentCity in cities:
                cities.remove(currentCity)


            if currentCity == -1:
                temp = self.shortest_path(path[-1], cities.pop())
                if temp[0] == INF:
                    return None
                path_to_rel = tuple(temp)
                path_to_rel = path_to_rel[0]
                path_to_rel.reverse()
                path_to_rel.pop()
                while path_to_rel:
                    t = path_to_rel.pop()
                    path.append(t)
                    if t in cities:
                        cities.remove(t)
                    if len(cities) == 0:
                        break
                # to_visit.remove(min(to_visit))
                continue


            if len(path) >= 2:
                if path[-2] == currentCity:
                    # path_to_rel = self.shortest_path(path[-1], cities.pop())
                    temp = self.shortest_path(path[-1], cities.pop())
                    if temp[0] == INF:
                        return None
                    path_to_rel = tuple(temp)
                    path_to_rel: list = path_to_rel[0]
                    path_to_rel.reverse()
                    path_to_rel.pop()
                    while path_to_rel:
                        t = path_to_rel.pop()
                        path.append(t)
                        if t in cities:
                            cities.remove(t)
                        if len(cities) == 0:
                            break
                    continue

            path.append(currentCity)
        return path

        """
        Return the cheapest neighbor of a current vertex.
        :param node: the current vertex's key.
        :return: the cheapest neighbor's key. -1 if there are no neighbors
        """
    def getCheapestNeighbour(self, node):
        neighbors = self.graphAlgo.all_out_edges_of_node(node)
        if len(neighbors) == 0:
            return -1
        cheapest = sys.float_info.max
        for neighbor in neighbors:
            cost = self.cost(node, neighbor)
            if cost < cheapest:
                cheapest = cost
                cheapestNeighbor = neighbor
        return cheapestNeighbor

    """
        TSP auxiliary functions
    """

    """
    Checks if the graph is connected or not.
    :return: True if yes, False else.
    """
    def is_connected(self) -> bool:
        vertices = self.graphAlgo.get_all_v()
        for vertex in vertices:
            if not self.BFS(vertex):
                return False
        return True

    """
    BFS algorithm implementation to check if graph is connected.
    That implementation is of GeeksForGeeks but improved and adapted to a direct graph.
    The source is in link https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/
    """
    def BFS(self, src: int) -> bool:
        # Mark all the vertices as not visited
        isVisited = [False] * len(self.graphAlgo.get_all_v())

        # Create a queue for BFS
        queue = []

        # Create a queue for the visited vertices
        visited = []

        # Mark the source node as visited and enqueue it
        queue.append(src)
        isVisited[src] = True
        visited.append(src)

        while len(queue) > 0:

            # Pop a vertex from queue
            currentNode = queue.pop(0)

            # Add currentNode's neighbors which were still not visited
            neighborsOut = self.graphAlgo.all_out_edges_of_node(currentNode)
            for neighbor in neighborsOut:
                if not isVisited[neighbor]:
                    queue.append(neighbor)
                    isVisited[neighbor] = True
                    visited.append(neighbor)
        return len(visited) == len(self.graphAlgo.get_all_v())










    """
    Plots the graph.
    If the nodes have a position, the nodes will be placed there.
    Otherwise, they will be placed in a random but elegant manner.
    @return: None
    """

    def plot_graph(self) -> None:
        self.runGui()
       # diagram_compare()

    def runGui(self):
        plt.figure(num=55, figsize=(8, 6), dpi=80, facecolor='tan')

        compList = self.connected_components()
        with plt.style.context('Solarize_Light2'):
            for i in range(len(compList)):
                comp = compList.pop(0)
                self.component_go(comp)
                plt.style.use('dark_background')
                self.draw_edges(list(self.graphAlgo.DirectedWeightedGraph.keys()))

        plt.title('DirectedWeightedGraph:(|V|=' + str(self.graphAlgo.v_size()) + ',' + '|E|= ' + str(
            self.graphAlgo.e_size()) + ')\n' + 'Yehiel Siri',
                  fontdict={'color': 'ivory', 'fontsize': 19, 'fontweight': 980})

        plt.show()

    def draw_edge(self, id1: int, id2: int) -> None:
        pos1 = self.graphAlgo.get_vertex(id1).getPosition()
        pos2 = self.graphAlgo.get_vertex(id2).getPosition()
        x1 = float(pos1[0])
        y1 = float(pos1[1])
        x2 = float(pos2[0])
        y2 = float(pos2[1])
        plt.arrow(x1, y1, 0.985 * (x2 - x1), 0.985 * (y2 - y1), length_includes_head=True, head_width=0.00038,
                  head_length=0.00045,
                  width=0.00001, fc='red', ec='black', zorder=1.1)

    def draw_node(self, id1: int) -> None:

        x1, y1 = self.graphAlgo.get_vertex(id1).getPosition()
        plt.scatter([x1], [y1])

    def component_go(self, idList) -> None:

        list_x = []
        list_y = []
        self.counterComp += 1
        label1 = 'In Component Number ' + str(self.counterComp)
        for n in idList:
            x, y = self.graphAlgo.get_vertex(n).getPosition()
            plt.annotate("Test",
                         xy=(0.9, 0.9), xycoords='data',
                         xytext=(0.8, 0.8), textcoords='data',
                         size=20, va="center", ha="center",
                         arrowprops=dict(arrowstyle="simple",
                                         connectionstyle="arc3,rad=-0.2"),
                         )
            list_x.append(float(x))
            list_y.append(float(y))
        plt.scatter(list_x, list_y, s=120, edgecolors='black', label=label1)

        plt.legend()

    def draw_edges(self, nodeList: list) -> None:
        dict1 = {}
        for n in nodeList:
            dict1 = self.graphAlgo.get_vertex(n).Ni_node_out
            for i in dict1:
                self.draw_edge(n, i)

    def connected_components(self) -> List[list]:

        result = []
        for i in self.graphAlgo.get_vertices():
            res = self.connected_component(i)
            if res not in result:
                result.append(res)
        return result

    """
    Returns a list of id's representing the vertices in the graph, in ascending order
    """
    def get_vertices1(self):
        ordered_vertices = []
        for vertex in self.graphAlgo.DirectedWeightedGraph:
            ordered_vertices.append(self.graphAlgo.DirectedWeightedGraph[vertex].id)
        return sorted(ordered_vertices)

    """
    Finds the Strongly Connected Component(SCC) that node id1 is a part of.
    @param id1: The node id
    @return: The list of nodes in the SCC
    """

    def connected_component(self, id1: int) -> list:
        visited = []
        answer = []
        # visiteDFS = []
        visiteDFS = self.DFS(id1)
        visiteDFS.remove(self.graphAlgo.get_vertex(id1))
        if len(self.graphAlgo.get_vertex(id1).Ni_node_out) == 0:
            return [id1]
        else:
            for node in visiteDFS:
                if node not in visited:

                    DFS_algo = self.DFS(node.id)

                    if self.graphAlgo.get_vertex(id1) in DFS_algo:
                        answer.append(node.id)
                    visited.append(node)

        answer.append(id1)

        answer.sort()

        return answer

    """
    DFS Algorithem
    @return: None
    """

    def DFS(self, start_node):
        visited = []
        stack = [self.graphAlgo.get_vertex(start_node)]
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.append(v)
                if v is not None:
                    for neighbor in v.Ni_node_out:
                        stack.append(self.graphAlgo.get_vertex(neighbor))
        return visited

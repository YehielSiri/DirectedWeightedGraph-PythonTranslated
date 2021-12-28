import copy
from unittest import TestCase
from DirectedWeightedGraph-PythonTranslated.src.GraphInterface import GraphInterface
from DirectedWeightedGraph-PythonTranslated.src.DiGraph import DiGraph
from DirectedWeightedGraph-PythonTranslated.src.GraphAlgo import GraphAlgo
from DirectedWeightedGraph-PythonTranslated.src.Node import Node
from typing import List
import os


class TestGraphAlgo(TestCase):

    def setUp(self) -> None:
        nodes = {}
        g = DiGraph(nodes)
        path = "C:\Users\YEHIEL\ObjectOrientedProgramminig_Ex3\data"
        self.graphAlgo = GraphAlgo(g)
        self.graphAlgo.load_from_json(path)

    def test_get_graph(self):
        tmp_DiGraph = self.graph_algo.get_graph()
        self.assertEqual(self.graph_algo.get_graph(), tmp_DiGraph)

    def test_load_from_json(self):
        file_loc = "C:\Users\YEHIEL\ObjectOrientedProgramminig_Ex3\data"
        self.assertEqual(self.graph_algo.load_from_json(file_loc), True)
        self.assertEqual(self.graph_algo.load_from_json("bla"), False)

    def test_save_to_json(self):
        file_loc = "C:\Users\YEHIEL\ObjectOrientedProgramminig_Ex3\data"
        self.graphAlgo.load_from_json(file_loc)
        self.graphAlgo.save_to_json("test_save")

    def test_shortest_path(self):
        print(self.graphAlgo.shortest_path(0, 6))

    def test_tsp(self):
        cities = {3, 2, 14, 5, 11, 10, 4}
        check = copy.deepcopy(cities)
        tsp = self.graphAlgo.TSP(cities)[0]
        print(tsp)

    def test_is_connected(self):
        self.assertTrue(self.graph_algo.is_connected())

    def test_center_point(self):
        print(self.graphAlgo.centerPoint())

    def test_plot_graph(self):
        self.graphAlgo.plot_graph()
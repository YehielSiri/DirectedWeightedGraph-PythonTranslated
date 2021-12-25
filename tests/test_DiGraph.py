from unittest import TestCase
from DiGraph import DiGraph
from src.Node import Node


class TestDiGraph(TestCase):

    def setUp(self) -> None:
        self.graphTest = DiGraph()
        for n in range(6):
            self.graphTest.add_node(n)
        self.graphTest.add_edge(0, 1, 1)
        self.graphTest.add_edge(1, 0, 1.1)
        self.graphTest.add_edge(1, 2, 1.3)
        self.graphTest.add_edge(2, 3, 1.1)
        self.graphTest.add_edge(1, 3, 1.9)
        self.graphTest.add_edge(3, 4, 8)
        self.graphTest.add_edge(4, 5, 1.9)
        self.graphTest.add_edge(5, 2, 1.9)

    def test_get_vertex(self):
        node = self.graphTest.Graph_DW[3]
        self.assertEqual(node, self.graphTest.get_vertex(3))

    def test_add_edge(self):
        y = self.graphTest.edge_size
        self.graphTest.add_edge(3, 0, 3.5)  # increase from 6 to 7
        x = self.graphTest.edge_size
        self.assertEqual(x, y + 1)

    def test_add_node(self):
        y = self.graphTest.num_vertices
        self.graphTest.add_node(3)
        self.graphTest.add_node(4)
        self.graphTest.add_node(7)
        self.graphTest.add_node(9)
        x = self.graphTest.num_vertices
        self.assertEqual(x, y + 2)

    def test_remove_edge(self):
        x = self.graphTest.edge_size
        self.graphTest.remove_edge(1, 0)
        z = self.graphTest.remove_edge(0, 3)
        self.assertFalse(z)
        self.graphTest.remove_edge(1, 3)
        y = self.graphTest.edge_size
        self.assertEqual(x - 2, y)

    def test_get_mc(self):
        x = self.graphTest.get_mc()
        self.graphTest.add_node(6)
        self.graphTest.add_edge(3, 6, 2.5)
        self.assertEqual(x + 2, self.graphTest.get_mc())

    def test_remove_node(self):
        y = self.graphTest.num_vertices
        self.graphTest.remove_node(0)
        x = self.graphTest.num_vertices
        self.assertEqual(x, y - 1)
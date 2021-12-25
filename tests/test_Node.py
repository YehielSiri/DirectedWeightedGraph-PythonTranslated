from unittest import TestCase
from Node import Node


class TestNode(TestCase):
    def setUp(self) -> None:
        pos = (0, 2, 0)
        self.nodeTest = Node(0, pos)
        # print(self.nodeTest)

    def test_get_id(self):
        self.assertEqual(self.nodeTest.id, 0)

    def test_add_out_ni(self):
        self.nodeTest.add_out_Ni(3, 4.5)
        x = self.nodeTest.Ni_node_out.pop(3)
        self.assertEqual(x, 4.5)

    def test_add_in_ni(self):
        self.nodeTest.add_in_Ni(2, 3.5)
        x = self.nodeTest.Ni_node_in.pop(2)
        self.assertEqual(x, 3.5)

    def test_get_weight_in(self):
        self.nodeTest.add_in_Ni(2, 3.5)
        self.assertEqual(self.nodeTest.get_weightIn(2), 3.5)

    def test_delete_ni_in(self):
        self.nodeTest.add_in_Ni(2, 3.5)

    def test_delete_ni_out(self):
        self.nodeTest.add_out_Ni(3, 4.5)
        self.assertEqual(self.nodeTest.Ni_node_out.__len__(), 1)
        self.nodeTest.deleteNi_out(3)
        self.assertEqual(self.nodeTest.Ni_node_out.__len__(), 0)

    def test_set_position(self):
        self.assertEqual(self.nodeTest.position, (0, 2, 0))
        posnew = (2, 5, 0)
        self.nodeTest.setPosition(posnew[0], posnew[1], posnew[2])
        self.assertEqual(self.nodeTest.position, posnew)

    def test_get_position(self):
        self.assertEqual(self.nodeTest.position, (0, 2, 0))
        posnew = self.nodeTest.getPosition()
        self.assertEqual(posnew, (0, 2, 0))
DirectedWeightedGraph.py

introduction
This repository implements the directed weighted graph in Python as the third assignment of Object Oriented Programming curse 
in Ariel University. In fact, this is the second assignment translated (from Java) to Python.
This project purpose is to compare between a Java code version to its Python translate. The Python version is this repository 
and the Java one is in link https://github.com/YehielSiri/DirectedWeightedGraph

How to install and run the project
firstlly, to clone the project into your local workspace, copy the commend below to the Bash git on your directory:
<git clone https://github.com/YehielSiri/DirectedWeightedGraph-PythonTranslated.git>
Then, execute the main.py file.

Data structures and classes details
Node class:
__init__(node, pos) - init the node; its position, its neighbors in & out etc.
__iter__() - return an iterator to the neighbors
__repr__() - return a string representation of the node by __str__() (an auxiliary function )
__str__() - return node as string
get_id() - return this node's key
get_weight(neighbor) - return the weight on the edge is which out of this node to the neighbor which is given
getNi() - return all the neighbors into this node
get_weightIn(neighbor) - return the weight on the edge is which in to this node from the neighbor which is given

DiGraph:


GraphAlgo:


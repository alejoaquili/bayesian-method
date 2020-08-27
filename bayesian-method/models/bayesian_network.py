from models.node import Node


class BayesianNetwork:
    def __init__(self):
        self.nodes = {}

    def generate_structure(self):
        rank_node = Node("rank")
        gre_node = Node("gre")
        gpa_node = Node("gpa")
        admit_node = Node("admit")
        self.add_asimetric_edge(gre_node, admit_node)
        self.add_asimetric_edge(gpa_node, admit_node)
        self.add_asimetric_edge(rank_node, gre_node)
        self.add_asimetric_edge(rank_node, gpa_node)
        self.nodes["admit"] = admit_node
        self.nodes["gre"] = gre_node
        self.nodes["gpa"] = gpa_node
        self.nodes["rank"] = rank_node

    def add_asimetric_edge(self, start_node, end_node):
        end_node.add_parent(start_node.name, start_node)
        start_node.add_child(end_node.name, end_node)

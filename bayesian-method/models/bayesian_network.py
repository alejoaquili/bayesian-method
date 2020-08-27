from models.node import Node


class BayesianNetwork:
    def __init__(self, data_relations):
        self.nodes = {}

    def generate_structure(self, data_relations):
        self.create_nodes(data_relations)
        for i in range(0, len(data_relations)):
            start_node = self.nodes[data_relations[i][0]]
            for j in range(0, len(data_relations[i][1])):
                end_node = self.nodes[data_relations[i][1][j]]
                self.add_asimetric_edge(start_node, end_node)

    def add_asimetric_edge(self, start_node, end_node):
        end_node.add_parent(start_node.name, start_node)
        start_node.add_child(end_node.name, end_node)

    def create_nodes(self, data_relations):
        for i in range(0, len(data_relations)):
            node_name = data_relations[i][0]
            self.nodes[node_name] = Node(node_name)

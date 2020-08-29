from models.node import Node


class BayesianNetwork:
    def __init__(self, data_relations, data_set, titles):
        self.nodes = {}
        self.generate_structure(data_relations)
        self.calculate_probabilities(data_set, titles)

    def generate_structure(self, data_relations):
        self.create_nodes(data_relations)
        for i in range(0, len(data_relations)):
            start_node = self.nodes[data_relations[i][0]]
            for j in range(0, len(data_relations[i][1])):
                end_node = self.nodes[data_relations[i][1][j]]
                self.add_asymmetric_edge(start_node, end_node)

    @staticmethod
    def add_asymmetric_edge(start_node, end_node):
        end_node.add_parent(start_node.name, start_node)
        start_node.add_child(end_node.name, end_node)

    def create_nodes(self, data_relations):
        for i in range(0, len(data_relations)):
            node_name = data_relations[i][0]
            self.nodes[node_name] = Node(node_name, data_relations[i][2])

    def calculate_probabilities(self, data_set, titles):
        for node_name in self.nodes.keys():
            current_node = self.nodes[node_name]
            current_node.generate_conditional_probabilities(data_set, titles)

    def calculate_total_probability(self, variables, values):
        pass

# P(admit = 0 | rank = 1) = dum valgre sum valgpaP(admit = 0, gre, gpa, rank = 1) / P(admit gre, gpa, rank = 1)
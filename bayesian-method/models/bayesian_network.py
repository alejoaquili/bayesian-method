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

    def calculate_total_generic_condicional_probability(self, variables, values, given_variables, given_values):
        numerator_probability = self.calculate_total_generic_probability(variables, values)
        denominator_probability =  self.calculate_total_generic_probability(given_variables, given_values)
        if numerator_probability is not None and denominator_probability is not None:
            return numerator_probability / denominator_probability
        return None

    def calculate_total_generic_probability(self, variables, values):
        specified_variables = []
        specified_values = []
        for i in range(0, len(variables)):
            if variables[i] in self.nodes:
                specified_variables.append(variables[i])
                specified_values.append([values[i]])
        for node_name in self.nodes:
            found = False
            for variable in variables:
                if variable == node_name:
                    found = True
            if not found:
                specified_variables.append(node_name)
                specified_values.append(self.nodes[node_name].values)
        return self.calculate_total_probability(specified_variables, specified_values)

    def calculate_total_probability(self, variables, values):
        keys = []
        self.generate_combinated_keys(variables, 0, len(variables), values, [], keys)
        total_probability = 0
        for key in keys:
            total_probability += self.calculate_term_probability(key)
        return total_probability

    def calculate_term_probability(self, key):
        variables_info = key.split(",")
        term_variables = []
        term_values = []
        for variable_info in variables_info:
            variable, value = variable_info.split("-")
            term_variables.append(variable)
            term_values.append(int(value))
        probability = 1
        for node_name in self.nodes:
            current_probability = self.nodes[node_name].get_conditional_probability(term_variables, term_values)
            if current_probability is not None:
                probability *= current_probability
        return probability

    def generate_combinated_keys(self, variables, current_index, size, values, values_for_variables, keys):
        if size == current_index:
            keys.append(Node.generate_key_with_values(variables, values_for_variables))
        else:
            for current_value in values[current_index]:
                values_for_variables.append(current_value)
                self.generate_combinated_keys(variables, current_index + 1, size,  values, values_for_variables, keys)
                del values_for_variables[-1]

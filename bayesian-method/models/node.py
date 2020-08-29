class Node:
    def __init__(self, name, values):
        self.name = name
        self.parents = {}
        self.children = {}
        self.values = values
        self.conditional_probabilities = {}

    def set_values(self, values):
        self.values = values

    def add_parent(self, parent_name, parent_node):
        self.parents[parent_name] = parent_node

    def add_child(self, child_name, child_node):
        self.children[child_name] = child_node

    def generate_conditional_probabilities(self, data_set, titles):
        parents_name_list = list(self.parents.keys())
        parents_name_list.append(self.name)
        parents_name_list.sort()
        all_variables = parents_name_list
        result_keys = []
        self.build_string_key(all_variables, len(all_variables), 0, "", result_keys)
        # print(result_keys)
        for row in range(0, len(data_set)):
            for key in result_keys:
                indexes = []
                values = []
                variables_info = key.split(",")
                for variable_info in variables_info:
                    variable, value = variable_info.split("-")
                    indexes.append(self.get_variable_index(variable, titles))
                    values.append(int(value))
                is_matching = True
                for i in range(0, len(indexes)):
                    if int(data_set[row][indexes[i]]) != int(values[i]):
                        is_matching = False
                if is_matching:
                    if key in self.conditional_probabilities:
                        self.conditional_probabilities[key] += 1
                    else:
                        self.conditional_probabilities[key] = 1
        for key in result_keys:
            matches = 0
            if key in self.conditional_probabilities:
                matches = self.conditional_probabilities[key]
            # TODO maybe add laplace corrector
            self.conditional_probabilities[key] = (matches + 1) / (len(data_set) + 2)
            # self.conditional_probabilities[key] = matches / len(data_set)

        # total = 0
        # for key in result_keys:
        #     if key in self.conditional_probabilities:
        #         total += self.conditional_probabilities[key]
        # print(total)

    def get_conditional_probability(self, term_variables, term_values):
        variables = []
        values = []
        for i in range(0, len(term_variables)):
            if self.name == term_variables[i]:
                variables.append(self.name)
                values.append(term_values[i])
        if len(variables) == 0:
            return None
        for i in range(0, len(term_variables)):
            current_variable = term_variables[i]
            if current_variable in self.parents:
                variables.append(current_variable)
                values.append(term_values[i])
        key = self.generate_key_with_values(variables, values)
        if key in self.conditional_probabilities:
            return self.conditional_probabilities[key]
        return None


    @staticmethod
    def get_variable_index(variable, titles):
        for i in range(0, len(titles)):
            if variable == titles[i]:
                return i
        return None

    @staticmethod
    def generate_key_with_values(variables, values):
        sorted_variables = variables.copy()
        sorted_variables.sort()
        key = ""
        for i in range(0, len(sorted_variables)):
            if i > 0:
                key += ","
            for j in range(0, len(variables)):
                if sorted_variables[i] == variables[j]:
                    key += "{variable}-{value}".format(variable=variables[j], value=values[j])
        return key

    def build_string_key(self, variables_name, size, current_index, current_key, result_keys):
        if current_index == size:
            return current_key
        if len(current_key) > 0:
            current_key += ","
        current_key += variables_name[current_index]
        values = self.values
        if variables_name[current_index] in self.parents:
            values = self.parents[variables_name[current_index]].values
        for current_value in values:
            new_key = "{previous}-{value}".format(previous=current_key, value=current_value)
            return_value = self.build_string_key(variables_name, size, current_index + 1, new_key, result_keys)
            if return_value is not None:
                result_keys.append(return_value)

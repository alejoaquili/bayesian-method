class Node:
    def __init__(self, name):
        self.name = name
        self.parents = {}
        self.children = {}
        self.conditional_probabilities = {}

    def add_parent(self, parent_name, parent_node):
        self.parents[parent_name] = parent_node

    def add_child(self, child_name, child_node):
        self.children[child_name] = child_node

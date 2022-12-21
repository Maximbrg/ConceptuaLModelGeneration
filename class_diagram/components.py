class entity:

    def __init__(self, label: str = None, node_type: str = None):
        """

        :param label:
        """
        self.label = label
        self.node_type = node_type


class relationship:

    def __init__(self, edge_type: str = None):
        """

        :param edge_type:
        """
        self.edge_type = edge_type

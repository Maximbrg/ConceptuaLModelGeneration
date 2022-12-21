import networkx as nx
from Framework.ModelElement.classDiagram import UCD as ucd
from class_diagram.components import entity


class simple_class_diagram:

    def __init__(self, graph: ucd = None):
        """

        :param graph:
        """
        self.class_diagram_graph = nx.DiGraph()

        # Add Nodes
        for node_key in graph.vertex_info.keys():
            node_label = graph.vertex_info[node_key]
            node_type = graph.vertex_type[node_key]
            self.class_diagram_graph.add_node(node_key, label=node_label, type=node_type)

        # Add Edges
        for edge_key in graph.edge_info.keys():
            edge_type = graph.edge_info[edge_key]
            self.class_diagram_graph.add_edge(edge_key[0], edge_key[1], type=edge_type)





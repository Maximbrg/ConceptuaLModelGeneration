import sys, glob, os
import pickle
import torch
from torch.utils.data import Dataset
from class_diagram.class_diagram import simple_class_diagram
from text_embedding.vectors import vectors
from torch.utils.data import DataLoader
import spacy
import networkx as nx

sys.path.append('../')


class CustomModelSetDataset(Dataset):
    def __init__(self, dataset_dir: str = None, nlp_type: str = None):
        """

        :param dataset_dir:
        :param nlp_type:
        """
        dataset_path = glob.glob(os.path.join(dataset_dir, 'class_diagrams.pickle'))[0]
        with open(dataset_path, 'rb') as handle:
            b = pickle.load(handle)

        if nlp_type == 'en_core_web_lg':
            self.nlp = spacy.load('en_core_web_lg')
        else:
            raise FileNotFoundError(f"Could not find expected configuration file for model " +
                                    f"under '{nlp_type}'")
        self.graph_node_features = {}
        self.graph_node_class = {}
        self.graph_edge_index = {}
        is_node_features_exists = False
        key = 0

        for graph_tuple in b:
            print(key)
            graph = graph_tuple[0]
            graph = simple_class_diagram(graph=graph)
            nodes = graph.class_diagram_graph.nodes
            edges = graph.class_diagram_graph.edges

            self.graph_node_features[key] = []
            self.graph_node_class[key] = []
            self.graph_edge_index[key] = {}

            self.graph_edge_index[key][0] = []
            self.graph_edge_index[key][1] = []

            self.output = {}

            # creating features to a node
            feature_path = glob.glob(os.path.join(dataset_dir + '\\text_embedding\\', 'dataset_1.pickle'))
            if len(feature_path) == 1 and not is_node_features_exists:
                with open(feature_path[0], 'rb') as handle:
                    b = pickle.load(handle)
                    self.graph_node_features = b
                    is_node_features_exists = True
            if not is_node_features_exists:
                node_features = vectors.create_vectors(nodes=nodes, nlp=self.nlp)
                self.graph_node_features[key].append(node_features)

            # type to a node matrix
            for node in nodes:
                self.graph_node_class[key].append(nodes[node]['type'])

            # adjacency matrix
            for edge in edges:
                self.graph_edge_index[key][0].append(edge[0])
                self.graph_edge_index[key][1].append(edge[1])

            key += 1  # next graph

        # save node features
        if not is_node_features_exists:
            path = dataset_dir + '\\text_embedding\\' + 'dataset_1.pickle'
            with open(path, 'wb') as handle:
                pickle.dump(self.graph_node_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.graph_node_features)

    def __getitem__(self, idx):
        return self.graph_node_features[idx], self.graph_edge_index[idx], self.graph_node_class[idx]


def run():
    file_path = 'C:\\Users\\max_b\\PycharmProjects\\ConceptuaLModelGeneration\\dataset_1'

    dataset = CustomModelSetDataset(
        dataset_dir=file_path, nlp_type='en_core_web_lg')
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)
    for i, train_data in enumerate(dataset):
        print(i, train_data[i].shape)


if __name__ == '__main__':
    run()

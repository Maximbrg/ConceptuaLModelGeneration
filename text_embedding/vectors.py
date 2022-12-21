import sys, glob, os

sys.path.append('../')
import pickle
import spacy
import pandas as pd


class vectors:

    def __init__(self, maps=0):
        self.vec = None
        self.maps = maps
        self.nlp = spacy.load('en_core_web_lg')

    @staticmethod
    def create_vectors(nodes: [] = None, nlp: object = None):
        """

        :param graph_id:
        :param dataset_dir:
        :param nodes:
        :param nlp:
        :return:
        """
        # node_features = glob.glob(os.path.join(dataset_dir + '\\text_embedding\\', str(graph_id) + '.pickle'))
        # if len(node_features) == 1:
        #     with open(node_features[0], 'rb') as handle:
        #         b = pickle.load(handle)
        #         return b

        # else:
        text_embeddings = []
        for key in nodes:
            token = nlp(str(nodes[key]['label']))
            text_embeddings.append(token.vector)
            # path = dataset_dir + '\\text_embedding\\' + str(graph_id) + '.pickle'
            # with open(path, 'wb') as handle:
            #     pickle.dump(text_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return text_embeddings

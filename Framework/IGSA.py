import random
import copy
import spacy
import warnings
warnings.filterwarnings("ignore", message=r"\[W008\]", category=UserWarning)


class search_igsa:

    def __init__(self, config):
        self.config = config
        self.nlp = spacy.load('en_core_web_lg')

    def greedy_algorithm_recursive(self, target_graph, query_graph):
        # if 'How to Pickle' in target_graph.vertex_info[-1]:
        #     print('gggg')
        th = self.config['th']
        k = self.config['k']
        hop = self.config['hop']
        open_list = []
        output_list = []
        close_list_query = [-1]
        is_visited = []
        similarity = self.similarity_estimation('-1', target_graph, query_graph)

        is_visited.append(similarity[0][1])
        open_list.append([(similarity[0][1], -1, similarity[0][0],
                           [similarity[0][1]])])  # (target_vertex, query_vertex, similarity, is_visted)

        while len(open_list) != 0:

            state = open_list.pop()

            query_id = state[len(state) - 1][1]
            neighbors = query_graph.neighbors(query_id)
            # print(len(open_list))
            # print(state)
            if not neighbors or query_graph.is_last(query_id):
                output_list.append(state)
                continue

            for vertex_query in neighbors:
                if abs(vertex_query) < abs(query_id):
                    continue
                close_list_query.append(vertex_query)
                is_above, similar_nodes, is_visited = self.next_similar_node(target_graph, query_graph, vertex_query,
                                                                        state[len(state) - 1][1], state[len(state) - 1][0],
                                                                        th,
                                                                        state[len(state) - 1][3], k, hop, is_visited)
                # next_similar_node(target_graph, query_graph, query_id, query_id_prev, target_id, th, is_visited_vertex)

                for vertex in similar_nodes:
                    cloned_state = copy.deepcopy(state)
                    if is_above:
                        cloned_state.append(vertex)
                        open_list.append(cloned_state)
                    elif len(vertex[3]) < 4:
                        # else:
                        cloned_state.append((vertex[0], state[len(state) - 1][1], vertex[2], vertex[3]))
                        open_list.append(cloned_state)

                    # else:
                    #     print()

        return output_list

    def next_similar_node(self, target_graph, query_graph, query_id, query_id_prev, target_id, th, is_visited_vertex, k, hop,
                          is_visited):
        max_similarity = []
        temp_similarity_calculation = []
        # print("next_similar_node", str(query_id))
        # query_text = query_graph.vertex_info[query_id].split()
        query_arc = query_graph.arc_type(query_id_prev, query_id)
        query_type = query_graph.get_type(query_id)
        for key in target_graph.neighbors(target_id):
            if key in is_visited:
                continue
            # target_text = target_graph.vertex_info[key].split()
            target_arc = target_graph.arc_type(target_id, key)
            target_type = target_graph.get_type(key)

            label_similarity = self.get_sim_between_2_nodes(target_graph, key, query_graph, query_id)
            edge_type_similarity = self.get_sim_between_2_edges(target_arc, query_arc)
            vertex_type_similarity = self.get_type_sim(target_type, query_type)

            similarity = (label_similarity + edge_type_similarity + vertex_type_similarity) / 3

            cloned_is_visited_vertex = copy.deepcopy(is_visited_vertex)
            cloned_is_visited_vertex.append(key)
            is_visited.append(key)
            max_similarity.append((key, query_id, similarity, cloned_is_visited_vertex))
            # print("[" + str(key) + "]: " + str(query_id), str(similarity))
        max_similarity.sort(key=lambda tup: tup[2])
        max_similarity.reverse()
        # print(query_id)
        if not max_similarity:
            # print([])
            return False, [], is_visited
        if max_similarity[0][2] > th:
            output = []
            counter = 0
            for item in max_similarity:
                if counter < k:
                    output.append(item)
                    counter += 1
            # print(output)
            return True, output, is_visited
        else:
            new_max_similarity = []
            for item in max_similarity:
                new_max_similarity.append((item[0], item[1], hop, item[3])) # hop item[2]
            # print(new_max_similarity)
            return False, new_max_similarity, is_visited

#
    def get_sim_between_2_nodes(self, target_graph, key, query_graph, query_id2):
        try:
            doc1 = self.config['vectors'][str(target_graph.vertex_info[key])]
        except:
            return 0
        doc2 = self.config['vectors'][query_graph.vertex_info[int(query_id2)]]
        try:
            sim = doc1.similarity(doc2)
        except:
            print()
        return sim


    def get_type_sim(self, type1, type2):
        return self.config['structure'].Similarity_matrix_class[type1][type2]


    def get_sim_between_2_edges(self, edge1, edge2):
        if edge2 in self.config['structure'].Similarity_matrix_edge[edge1].keys():
            return self.config['structure'].Similarity_matrix_edge[edge1][edge2]
        else:
            return self.config['structure'].Similarity_matrix_edge[edge2][edge1]

    def similarity_estimation(self, vertex_2, target_graph, query_graph):
        # print("Start finding the most similar vertex in the code graph to the first vertex of a query:")
        max_similarity = 0
        # node_id = vertex_2
        # query_text = query_graph.vertex_info[int(node_id)].split()
        # print(str(vertex_2))
        for key in target_graph.vertex_info:
            # node_text = target_graph.vertex_info[key].split()
            type_1 = query_graph.get_type(-1)
            type_2 = target_graph.get_type(key)
            # print("[" + str(key) + "]: " + str(vertex_2))

            sim_semantic = self.get_sim_between_2_nodes(target_graph, key, query_graph, vertex_2)

            sim_type = self.get_type_sim(type_1, type_2)

            sim = (sim_semantic + sim_type) / 2
            # print("similarity: " + str(sim))
            if sim > max_similarity:
                max_similarity = sim
                node_id = key

        return [(max_similarity, node_id)]


    def read_results(self, results):
        tot_output = []
        for item in results:
            output = []
            similarity = 0
            for step in item:
                output.append(step[0])
                similarity += step[2]

            # print(output, similarity / len(item))
            tot_output.append((output, similarity / len(item)))
        return tot_output


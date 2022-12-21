import copy
import spacy


class search_a_star_semantic:

    def __init__(self, config):
        self.config = config
        self.nlp = spacy.load('en_core_web_lg')

    def greedy_algorithm_recursive(self, target_graph, query_graph):
        th = self.config['th']
        k = self.config['k']
        hop = self.config['hop']
        open_list = []
        output_list = []
        is_visited = []
        similarity = self.similarity_estimation('-1', target_graph, query_graph)

        is_visited.append(similarity[0][1])
        # open_list.append([(similarity[0][1], -1, similarity[0][0],
        #                    [similarity[0][1]])])

        open_list.append(([similarity[0][1]], -1, similarity[0][0], [similarity[0][0]]))

        while len(open_list) != 0:
            open_list = sorted(open_list, key=lambda tup: tup[2])
            # open_list = Reverse(open_list)
            state = open_list.pop()

            query_id = state[1]
            neighbors = query_graph.neighbors(query_id)

            if not neighbors or query_graph.is_last(query_id):
                output_list.append(state)
                continue

            for vertex_query in neighbors:
                if abs(vertex_query) < abs(query_id):
                    continue
                is_above, similar_nodes, _ = self.next_similar_node_a(target_graph, query_graph, vertex_query,
                                                                 query_id, state[0][len(state[0]) - 1],
                                                                 th,
                                                                 is_visited, k, hop, is_visited)

                for vertex in similar_nodes:
                    next_node = vertex[0]
                    is_visited.append(next_node)
                    similarity = vertex[2]
                    if is_above:
                        query_node = vertex[1]
                    else:
                        query_node = query_id

                    similarities_path = copy.deepcopy(state[3])
                    similarities_path.append(similarity)

                    if is_above and query_graph.is_last(query_node):
                        path_estimation = self.pssEstimation(similarities_path, 4, True)
                        curr_state = copy.deepcopy(state[0])
                        curr_state.append(next_node)
                        open_list.append((curr_state, query_node, path_estimation, similarities_path))

                    else:
                        if is_above:
                            neighbors = query_graph.neighbors(query_id)
                            _, similar_nodes, _ = self.next_similar_node_a(target_graph, query_graph, neighbors[0],
                                                                      query_id, next_node,
                                                                      th,
                                                                      is_visited, k, hop, is_visited)
                        else:
                            _, similar_nodes, _ = self.next_similar_node_a(target_graph, query_graph, vertex_query,
                                                                      query_id, next_node,
                                                                      th,
                                                                      is_visited, k, hop, is_visited)

                        next_nodes_similarity = []

                        if similar_nodes == []:
                            continue

                        for item in similar_nodes:
                            next_nodes_similarity.append(item[2])

                        path_estimation = self.pssEstimation(similarities_path, 4, False, next_nodes=next_nodes_similarity)
                        curr_state = copy.deepcopy(state[0])
                        curr_state.append(next_node)
                        if path_estimation > hop:
                            open_list.append((curr_state, query_node, path_estimation, similarities_path))
        return output_list


    def next_similar_node_a(self, target_graph, query_graph, query_id, query_id_prev, target_id, th, is_visited_vertex, k,
                            hop,
                            is_visited):
        max_similarity = []
        temp_similarity_calculation = []
        # print("next_similar_node", str(query_id))
        query_text = query_graph.vertex_info[query_id].split()
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
            # is_visited.append(key)
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
            for item in max_similarity:
                output.append(item)
            # print(output)
            return True, output, is_visited
        else:
            new_max_similarity = []
            for item in max_similarity:
                new_max_similarity.append((item[0], item[1], item[2], item[3]))  # hop item[2]
            # print(new_max_similarity)
            return False, new_max_similarity, is_visited


    def pssEstimation(self, similarities, n, isFinal, next_nodes=[]):
        counter = 0
        weight_1 = 1
        for similarity in similarities:
            weight_1 *= similarity
            counter += 1

        if not isFinal:

            max_weight_2 = 0
            for curr_weight in next_nodes:
                if curr_weight > max_weight_2:  # and node[0] != node_path[i]:
                    max_weight_2 = curr_weight
            # print(weight_1, max_weight_2)
            pss_est = pow((weight_1 * max_weight_2), (1 / n))
            return pss_est

        else:
            return pow(weight_1, (1 / counter))

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

    def get_type_sim(self, type1, type2):
        return self.config['structure'].Similarity_matrix_class[type1][type2]

    def get_sim_between_2_edges(self, edge1, edge2):
        if edge2 in self.config['structure'].Similarity_matrix_edge[edge1].keys():
            return self.config['structure'].Similarity_matrix_edge[edge1][edge2]
        else:
            return self.config['structure'].Similarity_matrix_edge[edge2][edge1]

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

    def read_results(self, results):
        tot_output = []
        for item in results:
            tot_output.append((item[0], item[2]))
        return tot_output
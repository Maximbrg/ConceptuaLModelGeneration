import json
import copy



class me_map:

    def __init__(self, path):
        with open(path) as f:
           try:
            data = json.load(f)
           except:
               print()
        self.graph = {}
        self.vertex_info = {}
        self.vertex_type = {}
        self.vertex_vectors = {}
        self.edge_info = {}
        for node in data["nodeDataArray"]:
            self.graph[node["key"]] = []
            self.vertex_info[node["key"]] = node["text"]
            self.vertex_type[node["key"]] = node["category"]
            self.vertex_vectors[node["key"]] = ''

        for edge in data["linkDataArray"]:
            self.graph[edge["from"]].append(edge["to"])
            self.edge_info[(edge["from"], edge["to"])] = edge["text"]
        self.oldEdges = copy.deepcopy(self.edge_info)
        self.old_vertex_info = copy.deepcopy(self.vertex_info)
        self.old_vertex_type = copy.deepcopy(self.vertex_type)

        self.max_id = []
        for id in self.vertex_info.keys():
            is_entered = False
            for conected in self.graph[id]:
                if abs(id) < abs(conected):
                    is_entered = True
            if not is_entered:
                self.max_id.append(id)

    def is_last(self, id):
        if id in self.max_id:
            return True
        else:
            return False



    def num_of_vertices(self):
        return len(self.vertex_info)

    def neighbors(self, key):
        return self.graph[key]

    def arc_type(self, from_node, to_node):
        return self.edge_info[(from_node, to_node)]

    def vertex_text(self, key):
        return self.vertex_info[key]

    def get_first_node(self):
        return self.vertex_info[-1]

    def get_type(self, id):
        return self.vertex_type[id]

    def bfs(self):
        print(str(self.vertex_info))
        print(str(self.graph))
        queue = [-1]
        visited = {}
        for key in self.graph:
            visited[key] = False
        while queue:
            vertex = queue.pop(0)
            visited[vertex] = True
            print("ID:"+str(vertex)+" Text: "+self.vertex_info[vertex])
            print("My neighbor:")
            i = 0
            for key in self.graph[vertex]:
                count = count + len(key.split())
                i += 1
            #     if not visited[key]:
            #         queue.append(key)
            #     print("ID:"+str(key)+" Text: "+self.vertex_info[key])
            # print("--------------------------------------------------")
        print("AVG " + count /i )

    def update_query(self, graph):
        vertex_info = {} #id: 'label'
        vertex_type = {} #idL 'type'
        inverse_vertex_info = {} #{} 'lamel': id
        hashToName = {}
        graph2 = {}

        edge_info = {} #(id,id) : 'type'

        id = -1

        for key in graph.keys():
            try:
                vertex_info[id] = self.old_vertex_info[int(key)]
                vertex_type[id] = self.old_vertex_type[int(key)]
                inverse_vertex_info[vertex_info[int(id)]] = id
                value = graph[key][0]
                graph2[id] = []
                graph2[id].append(id - 1)

                edge_info[(int(id), int(id - 1))] = self.oldEdges[(int(id), int(value))]
                id -= 1

            except:
                break

        self.vertex_info = vertex_info
        self.vertex_type = vertex_type
        self.inverse_vertex_info = inverse_vertex_info
        self.graph = graph2
        self.edge_info = edge_info
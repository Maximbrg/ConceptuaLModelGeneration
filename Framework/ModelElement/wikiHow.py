import json
import copy


class wiki_how:

    def __init__(self, data):

        self.graph = {}
        self.vertex_info = {}
        self.vertex_type = {}
        self.vertex_vectors = {}
        self.inverse_vertex_info = {}
        self.edge_info = {}
        main_task = False
        id = -1
        if 'MainTask' in data.keys():
            main_task = True
            self.add_vertex(data['MainTask'], 'MainTask', id)
            id -= 1

        if 'Parts' in data.keys():
            for part in data['Parts']:
                self.add_vertex(part['PartName'], 'parts', id)
                self.addEdge(part['PartName'], data['MainTask'], 'non', twoSide=True)
                id -= 1

                for step in part['steps']:
                    self.add_vertex(step['Headline'], 'step', id)
                    self.addEdge(part['PartName'], step['Headline'], 'non', twoSide=True)
                    id -= 1

        if 'Methods' in data.keys():
            for part in data['Methods']:
                self.add_vertex(part['MethodName'], 'methods', id)
                self.addEdge(part['MethodName'], data['MainTask'], 'non', twoSide=True)
                id -= 1

                for step in part['Steps']:
                    self.add_vertex(step['Headline'], 'step', id)
                    self.addEdge(part['MethodName'], step['Headline'], 'non', twoSide=True)
                    id -= 1

        if 'Steps' in data.keys():
            for step in data['Steps']:
                self.add_vertex(step['Headline'], 'step', id)
                self.addEdge(data['MainTask'], step['Headline'], 'non', twoSide=True)
                id -= 1

        if 'Categories' in data.keys():
            for step in data['Categories']:
                self.add_vertex(step, 'category', id)

                self.addEdge(data['MainTask'], step, 'non', twoSide=True)
                id -= 1

        if 'Ingredients' in data.keys():
            for step in data['Ingredients']:
                self.add_vertex(step, 'Ingredient', id)
                self.addEdge(data['MainTask'], step, 'non', twoSide=True)
                id -= 1

        if 'Requirements' in data.keys():
            for step in data['Requirements']:
                self.add_vertex(step, 'Requirement', id)
                self.addEdge(data['MainTask'], step, 'non', twoSide=True)
                id -= 1

        if 'Tips' in data.keys():
            for step in data['Tips']:
                self.add_vertex(step, 'Tip', id)
                self.addEdge(data['MainTask'], step, 'non', twoSide=True)
                id -= 1


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

    def add_vertex(self, node1, type, id):
        self.vertex_info[id] = node1
        self.vertex_type[id] = type
        self.inverse_vertex_info[node1] = id
        self.graph[id] = []

    def update_vertex(self, id, type, name):
        prev_name = self.vertex_info[int(id)]
        self.vertex_info[int(id)] = name
        self.vertex_type[int(id)] = type
        self.inverse_vertex_info[name] = id
        del self.inverse_vertex_info[prev_name]


    def addEdge(self, node1, node2, type, twoSide=True):
        id1 = self.inverse_vertex_info[node1]
        id2 = self.inverse_vertex_info[node2]

        if id2 in self.graph[id1]:
            return

        self.graph[id1].append(id2)
        self.edge_info[(id1, id2)] = type

        if twoSide:
            self.graph[id2].append(id1)
            self.edge_info[(id2, id1)] = type

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
            print("ID:" + str(vertex) + " Text: " + self.vertex_info[vertex])
            print("My neighbor:")
            i = 0
            for key in self.graph[vertex]:
                count = count + len(key.split())
                i += 1
            #     if not visited[key]:
            #         queue.append(key)
            #     print("ID:"+str(key)+" Text: "+self.vertex_info[key])
            # print("--------------------------------------------------")
        print("AVG " + count / i)

    def update_query(self, graph):
        vertex_info = {}  # id: 'label'
        vertex_type = {}  # idL 'type'
        inverse_vertex_info = {}  # {} 'lamel': id
        hashToName = {}
        graph2 = {}

        edge_info = {}  # (id,id) : 'type'

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

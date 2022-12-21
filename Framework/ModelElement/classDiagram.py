import copy
import pyecore.ecore as Ecore
from pyecore.resources import ResourceSet, URI
import re
import os
import pickle
import time

class UCD:

    def __init__(self, path):
        id = -1

        self.graph = {}
        self.attributes = {}
        self.vertex_info = {}
        self.vertex_type = {}
        self.inverse_vertex_info = {}
        self.hashToName = {}
        self.vertex_vectors = {}
        self.edge_info = {}

        rset = ResourceSet()
        resource = rset.get_resource(URI(path))
        mm_root = resource.contents[0]
        classes = mm_root.eClassifiers

        for index, eclass in enumerate(classes):
            if 'EEnum' in str(type(eclass)):
                index -= 1
                continue
            class_type = 'Class'
            name = eclass.name

            if eclass.abstract:
                class_type = 'Abstract'
            elif eclass.interface:
                class_type = 'Interface'
            if any(ele.isupper() for ele in name):
                name = ' '.join(re.findall('[A-Z][^A-Z]*', name))
            # print(name, class_type)
            self.addVertex(name, class_type, id)

            self.attributes[id] = []

            for att in eclass.eAttributes:
                self.attributes[id].append(att.name)

            id -= 1

        for index, eclass in enumerate(classes):
            if 'EEnum' in str(type(eclass)):
                continue
            for superclasses in eclass.eSuperTypes.items:
                name_1 = eclass.name
                name_2 = superclasses.name
                if any(ele.isupper() for ele in name_1):
                    name_1 = ' '.join(re.findall('[A-Z][^A-Z]*', name_1))
                if any(ele.isupper() for ele in name_2):
                    name_2 = ' '.join(re.findall('[A-Z][^A-Z]*', name_2))

                self.addEdge(name_1, name_2, 'generalization')
                # print(name_1, name_2)
            for edges in eclass.eReferences:

                if edges.containment:
                 try:
                    name_1 = eclass.name
                    name_2 = edges.eType.name
                    if any(ele.isupper() for ele in name_1):
                        name_1 = ' '.join(re.findall('[A-Z][^A-Z]*', name_1))
                    if any(ele.isupper() for ele in name_2):
                        name_2 = ' '.join(re.findall('[A-Z][^A-Z]*', name_2))

                    self.addEdge(name_1, name_2, 'composite')
                    # print(name_1, name_2)
                 except:
                    continue
                else:
                    try:
                        name_1 = eclass.name
                        name_2 = edges.eType.name
                        if any(ele.isupper() for ele in name_1):
                            name_1 = ' '.join(re.findall('[A-Z][^A-Z]*', name_1))
                        if any(ele.isupper() for ele in name_1):
                            name_2 = ' '.join(re.findall('[A-Z][^A-Z]*', name_2))
                        self.addEdge(name_1, name_2, 'Association')
                        # print(name_1, name_2)
                    except:
                        continue
        self.oldEdges = copy.deepcopy(self.edge_info)
        self.old_vertex_info = copy.deepcopy(self.vertex_info)
        self.old_vertex_type = copy.deepcopy(self.vertex_type)
        self.old_inverse_info = copy.deepcopy(self.inverse_vertex_info)

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

    def addVertex(self, node1, type, id):
        self.vertex_info[id] = node1
        self.vertex_type[id] = type
        self.inverse_vertex_info[node1] = id
        self.graph[id] = []


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

    def get_first_node(self):
        return self.vertex_info[-1]

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
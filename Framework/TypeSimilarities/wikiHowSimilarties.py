class matrix_wh:

    def __init__(self):
        self.Similarity_matrix_edge = {'non': {}}
        self.Similarity_matrix_edge['non']['non'] = 1

        self.Similarity_matrix_class = {'MainTask': {}, 'methods': {}, 'parts': {}, 'step': {}, 'category': {},
                                     'Ingredient': {}, 'Requirement': {},
                                     'Tip': {}}

        self.Similarity_matrix_class['MainTask']['MainTask'] = 1
        self.Similarity_matrix_class['MainTask']['methods'] = 0
        self.Similarity_matrix_class['MainTask']['parts'] = 0
        self.Similarity_matrix_class['MainTask']['step'] = 0
        self.Similarity_matrix_class['MainTask']['category'] = 0
        self.Similarity_matrix_class['MainTask']['Ingredient'] = 0
        self.Similarity_matrix_class['MainTask']['Requirement'] = 0
        self.Similarity_matrix_class['MainTask']['Tip'] = 0

        self.Similarity_matrix_class['methods']['MainTask'] = 0
        self.Similarity_matrix_class['methods']['methods'] = 1
        self.Similarity_matrix_class['methods']['parts'] = 0.5
        self.Similarity_matrix_class['methods']['step'] = 0
        self.Similarity_matrix_class['methods']['category'] = 0
        self.Similarity_matrix_class['methods']['Ingredient'] = 0
        self.Similarity_matrix_class['methods']['Requirement'] = 0
        self.Similarity_matrix_class['methods']['Tip'] = 0

        self.Similarity_matrix_class['parts']['MainTask'] = 0
        self.Similarity_matrix_class['parts']['methods'] = 0.5
        self.Similarity_matrix_class['parts']['parts'] = 1
        self.Similarity_matrix_class['parts']['step'] = 0
        self.Similarity_matrix_class['parts']['category'] = 0
        self.Similarity_matrix_class['parts']['Ingredient'] = 0
        self.Similarity_matrix_class['parts']['Requirement'] = 0
        self.Similarity_matrix_class['parts']['Tip'] = 0

        self.Similarity_matrix_class['step']['MainTask'] = 0
        self.Similarity_matrix_class['step']['methods'] = 0
        self.Similarity_matrix_class['step']['parts'] = 0
        self.Similarity_matrix_class['step']['step'] = 1
        self.Similarity_matrix_class['step']['category'] = 0.5
        self.Similarity_matrix_class['step']['Ingredient'] = 0.5
        self.Similarity_matrix_class['step']['Requirement'] = 0.5
        self.Similarity_matrix_class['step']['Tip'] = 0.5

        self.Similarity_matrix_class['category']['MainTask'] = 0
        self.Similarity_matrix_class['category']['methods'] = 0
        self.Similarity_matrix_class['category']['parts'] = 0
        self.Similarity_matrix_class['category']['step'] = 0.5
        self.Similarity_matrix_class['category']['category'] = 1
        self.Similarity_matrix_class['category']['Ingredient'] = 0.5
        self.Similarity_matrix_class['category']['Requirement'] = 0.5
        self.Similarity_matrix_class['category']['Tip'] = 0.5

        self.Similarity_matrix_class['Ingredient']['MainTask'] = 0
        self.Similarity_matrix_class['Ingredient']['methods'] = 0
        self.Similarity_matrix_class['Ingredient']['parts'] = 0
        self.Similarity_matrix_class['Ingredient']['step'] = 0.5
        self.Similarity_matrix_class['Ingredient']['category'] = 0.5
        self.Similarity_matrix_class['Ingredient']['Ingredient'] = 1
        self.Similarity_matrix_class['Ingredient']['Requirement'] = 0.5
        self.Similarity_matrix_class['Ingredient']['Tip'] = 0.5

        self.Similarity_matrix_class['Requirement']['MainTask'] = 0
        self.Similarity_matrix_class['Requirement']['methods'] = 0
        self.Similarity_matrix_class['Requirement']['parts'] = 0
        self.Similarity_matrix_class['Requirement']['step'] = 0.5
        self.Similarity_matrix_class['Requirement']['category'] = 0.5
        self.Similarity_matrix_class['Requirement']['Ingredient'] = 0.5
        self.Similarity_matrix_class['Requirement']['Requirement'] = 1
        self.Similarity_matrix_class['Requirement']['Tip'] = 0.5

        self.Similarity_matrix_class['Tip']['MainTask'] = 0
        self.Similarity_matrix_class['Tip']['methods'] = 0
        self.Similarity_matrix_class['Tip']['parts'] = 0
        self.Similarity_matrix_class['Tip']['step'] = 0.5
        self.Similarity_matrix_class['Tip']['category'] = 0.5
        self.Similarity_matrix_class['Tip']['Ingredient'] = 0.5
        self.Similarity_matrix_class['Tip']['Requirement'] = 0.5
        self.Similarity_matrix_class['Tip']['Tip'] = 1



    def edge_matrix(self):
        return self.Similarity_matrix_edge

    def class_matrix(self):
        return self.Similarity_matrix_class
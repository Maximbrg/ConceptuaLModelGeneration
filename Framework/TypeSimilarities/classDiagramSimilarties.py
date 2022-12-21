class matrix_cd:

    def __init__(self):
        self.Similarity_matrix_edge = {'Association': {}, 'composite': {}, 'generalization': {}}

        self.Similarity_matrix_edge['Association']['Association'] = 1
        self.Similarity_matrix_edge['Association']['composite'] = 0.89
        self.Similarity_matrix_edge['Association']['generalization'] = 0.55

        self.Similarity_matrix_edge['composite']['composite'] = 1
        self.Similarity_matrix_edge['composite']['Association'] = 0.89
        self.Similarity_matrix_edge['composite']['generalization'] = 0.55

        self.Similarity_matrix_edge['generalization']['generalization'] = 1
        self.Similarity_matrix_edge['generalization']['Association'] = 0.51
        self.Similarity_matrix_edge['generalization']['composite'] = 0.51

        self.Similarity_matrix_class = {'Class': {}, 'Abstract': {}, 'Interface': {}}

        self.Similarity_matrix_class['Class']['Class'] = 1
        self.Similarity_matrix_class['Class']['Abstract'] = 0.75
        self.Similarity_matrix_class['Class']['Interface'] = 0.4

        self.Similarity_matrix_class['Abstract']['Class'] = 0.75
        self.Similarity_matrix_class['Abstract']['Abstract'] = 1
        self.Similarity_matrix_class['Abstract']['Interface'] = 0.75

        self.Similarity_matrix_class['Interface']['Class'] = 0.4
        self.Similarity_matrix_class['Interface']['Abstract'] = 0.7
        self.Similarity_matrix_class['Interface']['Interface'] = 1



    def edge_matrix(self):
        return self.Similarity_matrix_edge

    def class_matrix(self):
        return self.Similarity_matrix_class
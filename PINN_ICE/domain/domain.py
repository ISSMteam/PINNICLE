from ..parameters import DomainParameter
from ..utils import is_file_ext
import pandas as pd
import numpy as np
import deepxde as dde

class Domain:
    def __init__(self, parameters=DomainParameter()):
        self.parameters = parameters
        if is_file_ext(self.parameters.shapefile, '.exp'):
            self.vertices = self.get_polygon_vertices(self.parameters.shapefile)
            self.geometry = dde.geometry.Polygon(self.vertices)
        else:
            raise TypeError("File type in "+self.parameters.shapefile+" is currently not supported!")

    def get_polygon_vertices(self, filepath):
        """
        load exp domain file
        """
        df = pd.read_csv(filepath)

        domain_list = []
        for i in range(4, len(df)-1):
            # current vertex
            v = list(df.iloc[i])
            vertex = np.array(v[0].split(" "), dtype=float)
            vertex_list = list(vertex)
            # appending to main domain list
            domain_list.append(vertex_list)

        return domain_list

import pandas as pd
import numpy as np
from ..utils import *
import deepxde as dde

class domain:
    def __init__(self, path):
        if is_file_ext(path, '.exp'):
            self.vertices = self.get_polygon_vertices(path)
            self.domain = dde.geometry.Polygon(self.vertices)

    def get_polygon_vertices(self, filepath):
        """
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

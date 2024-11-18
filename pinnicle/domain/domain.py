from ..parameter import DomainParameter
from ..utils import is_file_ext
import pandas as pd
import numpy as np
import deepxde as dde

class Domain:
    def __init__(self, parameters=DomainParameter()):
        self.parameters = parameters
        # load space domain from shapefile
        if is_file_ext(self.parameters.shapefile, '.exp'):
            # create spatial domain
            self.vertices = self.get_polygon_vertices(self.parameters.shapefile)
            spacedomain = dde.geometry.Polygon(self.vertices)
        else:
            raise TypeError("File type in "+self.parameters.shapefile+" is currently not supported!")

        # create space-time domain
        if self.parameters.time_dependent:
            timedomain = dde.geometry.TimeDomain(self.parameters.start_time, self.parameters.end_time)
            self.geometry = dde.geometry.GeometryXTime(spacedomain, timedomain)
        else:
            self.geometry = spacedomain

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

    def inside(self, x):
        """
        return if given points are inside the domain
        """
        if self.parameters.time_dependent:
            # only check the spatial domain
            # TODO: add time domain
            return self.geometry.geometry.inside(x)
        else:
            return self.geometry.inside(x)

    def bbox(self):
        """
        return the bbox of the domain
        """
        if self.parameters.time_dependent:
            # only check the spatial domain
            # TODO: add time domain
            return self.geometry.geometry.bbox
        else:
            return self.geometry.bbox

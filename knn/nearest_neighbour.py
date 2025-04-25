import numpy as np


class NearestNeighbour:
    def __init__(self,k):
        self.k = k
        self.age = [10,15,17,19]
        self.height = [150,160,170,130]
        self.label = [0,1,0,1]

    @staticmethod
    def euclidian(x1, x2, y1, y2):
        dist_x = (x1 - x2) ** 2
        dist_y = (y2 - y1) ** 2
        return np.sqrt(dist_x + dist_y)

    def knn(self):
        distance = {}
        top_k = []
        new_point = (16,170)
        for i in range(len(self.age)):
            diff = self.euclidian(x1=self.age[i],x2=new_point[0],y1=new_point[1],y2=self.height[i])
            distance[diff] = self.label[i]
        for k in distance:
            top_k.append(distance[k])
        return top_k, distance


object = NearestNeighbour(k=3)
print(object.knn())




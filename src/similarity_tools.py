# -*- coding: utf-8 -*-
from geopy.distance import geodesic


class similarity(object):

    #坐标相似度
    def coordinate_sim(self, lat1, lon1, lat2, lon2):
        """
            球面距离的计算公式
        """
        dist = geodesic((lat1, lon1), (lat2, lon2)).km
        sim = 1 / dist
        return dist, sim

    #地址相似度
    def address_sim(self):
        pass

    #文本相似度
    def text_sim(self):
        pass


if __name__ == "__main__":
    a = similarity()
    d = a.coordinate_sim(39.9791, 116.483, 39.9834, 116.49)
    print(d)

# -*- coding: utf-8 -*-
import os
import pandas as pd
from numba import jit
from geopy.distance import geodesic
"""
1. 业务问题提出：如何融合不同源的POI数据
2. 业务问题转化和算法问题定义：在样本空间中P中搜索出（p1,p2）为同一个POI实体的数据对，
    以北京为例，假设高德，百度，谷歌的POI数各为50万，则样本空间为：50x50x3=7500万
3. 使用相似度算法进行数据过滤，减少样本空间
4. 人工标注
5. 模型训练、测试
6. 服务部署测试
"""
"""
一、经纬度距离换算

a）在纬度相等的情况下：
 经度每隔0.00001度，距离相差约1米；
 每隔0.0001度，距离相差约10米；
 每隔0.001度，距离相差约100米；
 每隔0.01度，距离相差约1000米；
 每隔0.1度，距离相差约10000米。
 
 b）在经度相等的情况下：
 纬度每隔0.00001度，距离相差约1.1米；
 每隔0.0001度，距离相差约11米；
 每隔0.001度，距离相差约111米；
 每隔0.01度，距离相差约1113米；
 每隔0.1度，距离相差约11132米。

"""
"""
1. 处理POI数据
2. 熟悉使用numba加速代码
"""

# "data/ROI_data/北京poi百度71万shp数据/北京.shp" 通过QGIS导出为 "data/ROI_data/BDbeijing71万.csv"
# pyshap 也可以读取，但不知道编码格式的情况下猜的让人有点蛋疼


#选取少部分数据用于调试代码
def get_demo_data(f1="data/ROI_data/baidu71万_format.csv", f2="data/ROI_data/gaode43万_format.csv"):

    baidu = pd.read_csv(f1)
    gaode = pd.read_csv(f2)

    chaoyang_baiddu = baidu.dropna()[baidu.dropna().Name.str.contains("招商银行")]
    chaoyang_baiddu.to_csv("data/ROI_data/baidu_test.csv", index=False)

    chaoyang_gaode = gaode.dropna()[gaode.dropna().Name.str.contains("招商银行")]
    chaoyang_gaode.to_csv("data/ROI_data/gaode_test.csv", index=False)


#通过三种相似度算法获取相似的POI实体对
def get_similar_pairs(f1="data/ROI_data/gaode_test.csv", f2="data/ROI_data/baidu_test.csv"):
    baidu = pd.read_csv(f2)
    gaode = pd.read_csv(f1)

    pairs = pd.DataFrame()
    for index, row in baidu.iterrows():

        for i, r in gaode.iterrows():
            dist = geodesic((row["Lat"], row["Lon"]), (r["Lat"], r["Lon"])).km
            if dist < 1:

                #if True:
                all_Series = pd.concat([r, row], axis=0, keys=['gaode', 'baidu'])
                pairs = pairs.append(all_Series, ignore_index=True)

    #pairs.to_csv("data/ROI_data/all_distance_min_1.csv", index=False)

    #writer = pd.ExcelWriter('/Users/majun/Documents/临时文件/output.xlsx', index=False)
    #pairs.to_excel(writer)
    pairs.to_excel('/Users/majun/Documents/临时文件/output.xlsx', index=False)


# 数据属性统一
def attr_format(f1="data/ROI_data/BDbeijing71万.csv", f2="data/ROI_data/GDbeijing47万.csv"):

    baidu = pd.read_csv(f1)
    gaode = pd.read_csv(f2)

    # 统一字段
    baidu = baidu[["Name", "Address", "Lon", "Lat", "BdSmallTag", "Area", "Tel"]]
    gaode = gaode[["name", "addr", "locationx", "locationy", "type", "district", "tel"]]

    baidu.rename(columns={
        "Name": "Name",
        "Address": "Address",
        "Lon": "Lon",
        "Lat": "Lat",
        "BdSmallTag": "Tag",
        "Area": "Area",
        "Tel": "Tel",
    }, inplace=True)
    gaode.rename(columns={
        "name": "Name",
        "addr": "Address",
        "locationx": "Lon",
        "locationy": "Lat",
        "type": "Tag",
        "district": "Area",
        "tel": "Tel",
    }, inplace=True)

    baidu.to_csv("data/ROI_data/baidu71万_format.csv", index=False)
    gaode.to_csv("data/ROI_data/gaode43万_format.csv", index=False)


if __name__ == "__main__":
    #attr_format()
    #get_demo_data()
    get_similar_pairs()

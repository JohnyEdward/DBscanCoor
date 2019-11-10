import pandas as pd
from math import radians
from math import tan, atan, acos, sin, cos, asin, sqrt
import itertools
from sklearn.cluster import DBSCAN
import numpy as np


# https://blog.csdn.net/qq_32618817/article/details/81172103
# https://blog.csdn.net/jackaduma/article/details/52734731
# https://www.biaodianfu.com/dbscan.html
# 该函数是为了通过经纬度计算两点之间的距离（以米为单位）
def geodistance(array_1,array_2):
    lng1 = array_1[0]
    lat1 = array_1[1]
    lng2 = array_2[0]
    lat2 = array_2[1]
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    dis=2*asin(sqrt(a))*6371*1000
    return dis


def main():
    data = pd.read_csv('outlier_data.csv', sep=',')
    data.columns = ['id', 'time', 'lon', 'lat', 'last_lon', 'last_lat', 'distance'] # 原来是中文的列名，担心之后的解码问题，更改了列名
    data = data[['id', 'time', 'lon', 'lat']] # 只需要其中的四列数据

    # 该部分是去掉那些上传数据次数小于3的车辆id
    count = data.groupby(by=['id'])['time'].agg({'count': len}).reset_index()# 算出每一个车辆id收集的数据次数
    data = pd.merge(data, count, on=['id'], how='left')
    data['lon_lat'] = data.apply(lambda x: [x['lon'], x['lat']], axis=1)# 为了方便使用geodistance这个经纬度换算公式，更改一下数据格式
    data = data[data['count'] > 3]

    # 对于清洗之后的数据，对每一个groupby的group进行处理，将DBSCAN之后的label值赋予成新的一列
    new = pd.DataFrame()
    for id_index,group in data.groupby(by=['id']):
        dbscan = DBSCAN(eps=500, min_samples=4, metric=geodistance).fit(list(group['lon_lat']))# 对于DBSCAN来说，两个最重要的参数就是eps，和min_samples。当然这两个值不是随便定义的，这个在下文再说
        group['label'] = dbscan.labels_
        new = new.append(group)# 我在这里被坑得不浅，也留在下文总结
    print(new)

    clean_data = new[new['label'] != -1]# label为-1的即为离群点，删除掉离群点
    # 然后求出其他点的中心点
    temp_lon = clean_data.groupby(by=['id'])['lon'].mean().reset_index()
    temp_lat = clean_data.groupby(by=['id'])['lat'].mean().reset_index()
    result = pd.merge(temp_lon,temp_lat,on=['id'],how='left')
    print(result.head(20))
    result.to_csv('去除离散点的数据.csv')
    return result


if __name__ == '__main__':
    main()
import scipy.sparse as sparse
import numpy as np
import random
import csv
from haversine import haversine, haversine_vector
def get_region(place_coords,size,path):
    """
    place_coords: list, index:poi_id value:(poi_lat, poi_lng)
    size: int, region size(m)
    """
    latitude_max = -55000
    latitude_min =55000
    longitude_max = -55000
    longitude_min = 55000
    places = []
    for lid in range(len(place_coords)): # lat, long의 min, max 구하기
        now_lat = place_coords[lid][0]
        now_lng = place_coords[lid][1]
        latitude_min = min(latitude_min, now_lat)
        latitude_max = max(latitude_max, now_lat)
        
        longitude_min = min(longitude_min, now_lng)
        longitude_max = max(longitude_max, now_lng)
        places.append((lid,now_lat,now_lng))
    
    print("lat_max:{latmax} lat_min:{latmin}".format(latmax=latitude_max, latmin=latitude_min))
    print("lng_max:{lngmax} lng_min:{lngmin}".format(lngmax=longitude_max, lngmin=longitude_min))

    #사다리꼴 모양에서 윗변, 아랫변, 좌우 높이 구하기
    width1 = haversine((latitude_max,longitude_max),(latitude_max,longitude_min), unit="m")
    width2 = haversine((latitude_min,longitude_max),(latitude_min,longitude_min), unit="m")
    
    height1 = haversine((latitude_max,longitude_max),(latitude_min,longitude_max), unit="m")
    height2 = haversine((latitude_max,longitude_min),(latitude_min,longitude_min), unit="m")

    #지역 쪼개기
    colnum = int((width2+width1)/2/size)
    rownum = int(height1/size)
    print("row: {} col: {}".format(rownum, colnum))
    alpha = (latitude_max - latitude_min)/rownum # 가로 한칸단 좌표 크기
    delta = (longitude_max - longitude_min)/colnum # 세로 한칸당 좌표 크기
    areaRangeArr = np.zeros((rownum, colnum, 4)) # 지역 임베딩을 매트릭스로 표현
    areaBusinessArr = dict()
    for i in range(rownum): # 각 매트릭스 인덱스별로 dict형식의 lat,long 을 저장하기 위한 dict 선언
        areaBusinessArr[i]=dict()
        for j in range(colnum):
            areaBusinessArr[i][j]=dict()

    place_region = np.zeros([len(place_coords)])-1 #지역 임베딩을 저장할 array
    for i in range(rownum):
        print("row{}".format(i))
        lat_min = latitude_min + alpha * (i) # lat 한칸의 시작 좌표, 위 delta처럼 alpha?로 지정해서 계산해도 될듯 ?
        lat_max = latitude_min + alpha * (i+1) # lat 한칸의 끝나는 좌표
        target = [x for x in places if x[1]>=lat_min and x[1]<=lat_max] # poi 좌표들 중 위 latitude 조건에 만족하는 poi들의 모음
        for j in range(colnum):
            lng_min = longitude_min + delta * j # long 한칸의 시작 좌표
            lng_max = longitude_min + delta * (j+1) #long 한칸의 끝나는 좌표

            # 구역을 나누는 좌표 저장
            areaRangeArr[i][j][0] = lat_min
            areaRangeArr[i][j][1] = lat_max
            areaRangeArr[i][j][2] = lng_min
            areaRangeArr[i][j][3] = lng_max

            for lid in target: # 각 poi에 대하여, 이때 poi는 위 lat 좌표 안에 있는 poi들
                if place_region[lid[0]]>=0: # 이미 region이 정해진 지역에 대해서는 pass
                    continue
                
                now_lat = place_coords[lid[0]][0] # 여기 바로 lid[1] 해도 될듯 ?
                now_lng = place_coords[lid[0]][1]
                if (now_lng<lng_max ) and (now_lat < lat_max):
                    place_region[lid[0]] = colnum*i + j 
                elif j == colnum-1 and i == rownum-1: # 맨 마지막 좌표일 때 (20,20)
                    if now_lng<=lng_max and now_lat <= lat_max: # 안쪽에 걸친다면
                        place_region[lid[0]] = colnum*i + j # 
                elif j == colnum-1:
                    if now_lng<=lng_max and now_lat < lat_max:
                        place_region[lid[0]] = colnum*i + j
                elif i == rownum-1:
                    if now_lng < lng_max and now_lat <= lat_max:
                        place_region[lid[0]] = colnum*i + j
                
    f= open(path+"poi_region.txt","w")
    for i in range(len(place_region)):
        f.write("{}\t{}\n".format(i,int(place_region[i])))
    f.close()

def train_test_split_with_time(place_list, freq_list, time_list, test_size):
    li = []
    for i in range(len(place_list)):
        li.append((place_list[i],time_list[i], freq_list[i]))
    
    li.sort(key=lambda x:-x[1])
    test = li[:int(len(li)*test_size)]
    train = li[int(len(li)*test_size):]
    random.shuffle(train)

    test_place = []
    test_freq = []
    for i in test:
        test_place.append(i[0])
        test_freq.append(i[2])

    train_place=[]
    train_freq=[]
    for i in train:
        train_place.append(i[0])
        train_freq.append(i[2])

    return train_place, test_place, train_freq, test_freq
def train_test_val_split_with_time(place_list, freq_list, time_list, test_size, val_size):
    li = []
    for i in range(len(place_list)):
        li.append((place_list[i], time_list[i], freq_list[i]))
    li.sort(key=lambda x:-x[1])
    test = li[:int(len(li)*test_size)]
    train_ = li[int(len(li)*test_size):]

    val_num = int(len(li)*val_size)
    if val_num == 0:
        val_num=1
    val = train_[:val_num]
    train = train_[val_num:]

    random.shuffle(train)
    test_place = []
    test_freq = []
    for i in test:
        test_place.append(i[0])
        test_freq.append(i[2])

    train_place=[]
    train_freq=[]
    for i in train:
        train_place.append(i[0])
        train_freq.append(i[2])

    val_place = []
    val_freq = []

    for i in val:
        val_place.append(i[0])
        val_freq.append(i[2])
    return train_place, test_place, val_place, train_freq, test_freq, val_freq
def get_region_num(path):
    input_file = 'poi_region.txt'
    output_file = 'poi_region_sorted.txt'


    # CSV 파일을 읽고 데이터를 리스트로 저장
    data = []
    with open(path+input_file, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
    #data.append(header)
        for row in csv_reader:
            row = [int(row[0]), int(row[1])]
            data.append(row)

    data.sort(key=lambda x: x[1])  # region 기준으로 데이터를 정렬

    idx = 0 # region을 0번부터 매핑시키기 위한 새로운 리스트 생성
    before_region_id = data[0][1] # 맨 첫번째 buid 기준
    new_data = []
    for i in data: # 모든 데이터에 대하여
        if i[1] == before_region_id: #buid가 같다면
            new_data.append([i[0], idx]) 
        else:
            idx += 1
            before_region_id = i[1]
            new_data.append([i[0], idx])

    new_data.sort(key=lambda x: x[0])  # user_id 순으로 정렬
    ma = max(new_data, key=lambda x:x[1])
    # 정렬된 데이터를 새로운 파일에 저장
    with open(path+output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')
        csv_writer.writerows(new_data)

    print("region num: {}".format(ma[1]+1))
    return ma[1]+1
def train_test_val_split(place_list, freq_list, test_size, val_size):
    li = []
    for i in range(len(place_list)):
        li.append((place_list[i], freq_list[i]))
    
    random.shuffle(li)
    test = li[:int(len(li)*test_size)]
    train_ = li[int(len(li)*test_size):]
    val_num = int(len(li)*val_size)
    if val_num == 0:
        val_num=1
    val = train_[:val_num]
    train = train_[val_num:]

    test_place = []
    test_freq = []
    for i in test:
        test_place.append(i[0])
        test_freq.append(i[1])

    train_place=[]
    train_freq=[]
    for i in train:
        train_place.append(i[0])
        train_freq.append(i[1])

    val_place = []
    val_freq = []

    for i in val:
        val_place.append(i[0])
        val_freq.append(i[1])
    return train_place, test_place, val_place, train_freq, test_freq, val_freq

class Dataset(object):
    def __init__(self,user_num,_poi_num,directory_path):
        self.user_num = user_num
        self.poi_num = _poi_num
        self.directory_path = directory_path
        self.checkin_file = 'checkins.txt'
        self.poi_file = 'poi_coos.txt'
    def read_raw_data(self):
        all_data = open(self.directory_path + self.checkin_file, 'r').readlines()
        sparse_raw_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        sparse_raw_time_matrix= sparse.dok_matrix((self.user_num, self.poi_num))
        for eachline in all_data:
            uid, lid, time = eachline.strip().split()
            uid, lid, time = int(uid), int(lid), float(time)
            sparse_raw_matrix[uid, lid] = sparse_raw_matrix[uid, lid] + 1
            if sparse_raw_time_matrix[uid,lid] < time:
                sparse_raw_time_matrix[uid, lid] = time
        return sparse_raw_matrix.tocsr(), sparse_raw_time_matrix.tocsr()

    def split_data(self, raw_matrix, time_matrix, random_seed=0):
        test_size = 0.2
        val_size = 0.1
        train_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        
        val_positive = []
        test_positive = []
        self.POI_POI_Graph = np.zeros([self.poi_num,self.poi_num])
        self.user_POI_Graph = np.zeros([self.user_num,self.poi_num])
        
        pois = set(range(self.poi_num))
        for user_id in range(self.user_num):
            place_list = raw_matrix.getrow(user_id).indices
            freq_list = raw_matrix.getrow(user_id).data
            time_list = time_matrix.getrow(user_id).data

            train_place, test_place, val_place, train_freq, test_freq, val_freq = train_test_val_split_with_time(place_list, freq_list, time_list, test_size, val_size)
            
            for i in range(len(train_place)):
                train_matrix[user_id, train_place[i]] = train_freq[i]

                self.user_POI_Graph[user_id][train_place[i]] = 1

                if i <len(train_place)-1:
                    self.POI_POI_Graph[train_place[i]][train_place[i+1]] +=1
            test_positive.append(test_place)
            val_positive.append(val_place)

            # train_negative.append(negative[int(len(negative)*test_size):])
            # ln = len(negative[int(len(negative)*test_size):])
            # test_ln = len(negative[:int(len(negative)*test_size)])
            # test_negative.append(negative[:test_ln])
            # val_negative.append(negative[test_ln:test_ln + int(ln*val_size)])
        # sparse.save_npz('./data/Foursquare/train_matrix.npz', train_matrix)

        return train_matrix.tocsr(), test_positive, val_positive

    def read_poi_coos(self,near_POI_num):
        poi_coos = {}
        poi_data = open(self.directory_path + self.poi_file, 'r').readlines()
        for eachline in poi_data:
            lid, lat, lng = eachline.strip().split()
            lid, lat, lng = int(lid), float(lat), float(lng)
            poi_coos[lid] = (lat, lng)

        place_coords = []
        for k, v in poi_coos.items():
            place_coords.append([v[0], v[1]])
        self.place_coos = place_coords
        self.dist_matrix = np.array(haversine_vector(place_coords,place_coords,comb=True))
        self.nearPOI = np.argpartition(self.dist_matrix,near_POI_num)[:,:near_POI_num]
        
        return place_coords

    def generate_data(self, random_seed=0, near_POI_num = 50):
        raw_matrix, time_matrix = self.read_raw_data()
        train_matrix, test_positive, val_positive = self.split_data(raw_matrix, time_matrix, random_seed)
        place_coords =self.read_poi_coos(near_POI_num)
        # a = train_matrix.toarray()
        # self.near_list = []
        # for i in range(len(a[0])):
        #     t = np.where(self.dist_matrix[i]<1.0)
        #     na=a[:,t].reshape(len(a),-1)
        #     norms = np.linalg.norm(na,axis=0).reshape(1,-1).repeat(len(t),axis=0)
        #     sim = np.dot(na.T,na)/(norms*norms.T+0.000000001)
        #     self.near_list.append(np.argpartition(sim,-50)[:,len(self.dist_matrix)-50:])


        # ext = 1/(np.exp(self.dist_matrix)*10)
        
        # na = np.repeat(np.linalg.norm(a,axis=0).reshape(1,-1),len(self.dist_matrix),axis=0)
        # sim = np.dot(a.T,a)/((na*na.T)+0.0000000001)+ext
        # self.nearPOI = np.argpartition(sim,-50)[:,len(self.dist_matrix)-50:]
        # self.nearPOI = np.argpartition(self.dist_matrix,50)[:,:50]
        return train_matrix, test_positive, val_positive, place_coords
def cosine_sim(A,B):
    return np.dot(A,B)/((np.linalg.norm(A)*np.linalg.norm(B))+0.0000000001)
if __name__ == '__main__':
    # train_matrix, test_positive, test_negative, val_positive, val_negative, place_coords= Dataset(9902,6427,"./data/philadelphia_downtown/").generate_data()
    dataset = Dataset(3725,10768,"./data/Tokyo/")
    train_matrix, test_positive, val_positive, place_coords= dataset.generate_data()
    nearPOI_user_matrix = []
    nearPOI = []
    a = train_matrix.toarray()
    ext = 1/(np.exp(dataset.dist_matrix)*10)
    
    na = np.repeat(np.linalg.norm(a,axis=0).reshape(1,-1),len(dataset.dist_matrix),axis=0)
    nt = na.T
    sim = np.dot(a.T,a)/((na*na.T)+0.0000000001)+ext
    poi = np.argpartition(sim,-50)[:,len(dataset.dist_matrix)-50:]
    for i in range(len(dataset.dist_matrix)):
        indices = np.arange(len(dataset.dist_matrix))
        # indices = np.where(dataset.dist_matrix[i] < 10.0)[0]
        # ext = 1/np.exp(dataset.dist_matrix[i][indices])
        t = a[:,indices]
        t = t.T
        origin = a[:,i]
        origin = np.repeat(origin.reshape([1,-1]),len(indices),axis=0)
        sim = np.zeros(len(origin))
        for j in range(len(origin)):
            sim[j] = cosine_sim(origin[j],t[j])+ext[j]
        nearPOI.append(np.argpartition(sim,-50)[len(origin)-50:])

    for i in range(len(dataset.nearPOI)):
        nearPOI_user_matrix.append(a[:,dataset.nearPOI[i]])
    print(train_matrix.shape, len(test_positive), len(place_coords))

    


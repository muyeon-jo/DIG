import datasets
from haversine import haversine
def user_visit_distance_average(dataset,raw_matrix,place_coords):
    user_num = dataset.user_num
    item_num = dataset.poi_num
    user_visit_distances = []
    user_visit_distances_std = []
    for uid in range(user_num):
        history = raw_matrix.getrow(uid).indices.tolist()
        a = 0
        for pid in history:
            p_pos = place_coords[pid]
            s = 0
            for i in history:
                s+=haversine(p_pos,place_coords[i])
            s /= (len(history)-1)
            a+=s
        user_visit_distances.append(a/len(history))

        a = 0
        for pid in history:
            p_pos = place_coords[pid]
            s = 0
            for i in history:
                if(i != pid):
                    s+=(haversine(p_pos,place_coords[i])-user_visit_distances[uid])**2
            s /= (len(history)-1)
            a+=s**(0.5)
        user_visit_distances_std.append(a/len(history))
    distance_average = sum(user_visit_distances)/user_num
    distance_std = sum(user_visit_distances_std)/user_num
    with open("./experiment/visit_distance_average.txt","w") as f:
        f.write("dataset: {}\nuser_num{}\nitem_num{}\n".format(dataset.directory_path,user_num,item_num))
        f.write("user_visit_distance_average\n")
        for uid in range(user_num):
            f.write("{}\t{}\n".format(uid,user_visit_distances[uid]))
        f.write("\naverage\n")
        f.write("{}".format(distance_average))
    
    with open("./experiment/visit_distance_std.txt","w") as f:
        f.write("dataset: {}\nuser_num{}\nitem_num{}\n".format(dataset.directory_path,user_num,item_num))
        f.write("user_visit_distance_std\n")
        for uid in range(user_num):
            f.write("{}\t{}\n".format(uid,user_visit_distances_std[uid]))
        f.write("\naverage\n")
        f.write("{}".format(distance_std))
if __name__ == '__main__':
    dataset = datasets.Dataset(6638,21102,"data/NewYork/")
    raw_matrix, time_matrix = dataset.read_raw_data()
    place_coords =dataset.read_poi_coos()
    user_visit_distance_average(dataset,raw_matrix,place_coords)

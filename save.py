import numpy as np
def save_experiment_result(directory, results,k_list,epoch):
    f=open(directory+"/results.txt","w")
    recall = results[0]
    precision = results[1]
    hitrate = results[2]

    f.write("epoch:{}\n".format(epoch))
    for i in range(len(k_list)):
        f.write(str(k_list[i])+"\t")
    f.write("\n")
    for i in range(len(recall)):
        f.write(str(recall[i])+"\t")
    f.write("\n")
    for i in range(len(precision)):
        f.write(str(precision[i])+"\t")
    f.write("\n")
    for i in range(len(hitrate)):
        f.write(str(hitrate[i])+"\t")
    f.write("\n")
    f.close()

def save_intersection(directory, results,num_items):
    ingoing = results[0]
    outgoing = results[1]
    f1=open(directory+"/ingoing_intersection.txt","w")
    for li in ingoing.indices.tolist():
        f1.write(str(li)+"\n")
    f1.close()

    f1=open(directory+"/outgoing_intersection.txt","w")
    for li in outgoing.indices.tolist():
        f1.write(str(li)+"\n")
    f1.close()

    f1=open(directory+"/intersection.txt","w")
    sum = 0
    res = []
    for li in range(len(outgoing.indices.tolist())):
        a = outgoing.indices[li].detach().cpu().numpy()
        b = ingoing.indices[li].detach().cpu().numpy()
        inter = np.intersect1d(a,b)
        sum+=len(inter)
        res.append(len(inter))
        f1.write(str(len(inter))+" "+str(inter)+"\n")
    f1.write("sum: "+str(sum)+"\n")
    f1.write("avr: "+str(sum/num_items)+"\n")
    f1.write("var: "+str(np.var(res))+"\n")
    f1.write("std: "+str(np.std(res))+"\n")
    f1.close()
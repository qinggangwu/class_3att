
import matplotlib.pyplot as plt
import numpy as np

def show_log(logTxt_paht):

    with open(logTxt_paht,'r') as f:
        loginfo = f.readlines()

    Lossor = [ii.split() for ii in loginfo  if 'Loss: ' in ii and len(ii) > 165]  # float(ii_nei.split()[7])   for ii_nei in ii
    # print(len(Lossor[100]))
    print(Lossor)  # ;quit()

    Loss = [float(ii.split()[7]) for ii in loginfo  if 'Loss: ' in ii and len(ii) > 165]  # loss
    avg_loss = [float(ii.split()[9][:-1]) for ii in loginfo if 'avg_loss: ' in ii and len(ii) > 165]  # loss
    dit_acc = [float(ii.split()[11][2:]) for ii in loginfo if 'd:' in ii and len(ii) > 165]  # 朝向正确率
    type_acc = [float(ii.split()[12][2:]) for ii in loginfo if 't:' in ii and len(ii) > 165]  # 类型正确率
    color_acc = [float(ii.split()[13][2:]) for ii in loginfo if 'c:' in ii and len(ii) > 165]  # 颜色正确率

    x = np.array([ i*10 for i in range(len(Loss)) ])
    Loss = np.array(Loss)
    avg_loss = np.array(avg_loss)
    dit_acc = np.array(dit_acc)
    type_acc = np.array(type_acc)
    color_acc = np.array(color_acc)

    plt.plot(x, Loss, linewidth=1)
    plt.plot(x, avg_loss, linewidth=1)

    plt.title("Loss", fontsize=24)  # 标题及字号
    plt.xlabel("iter", fontsize=14)  # X轴标题及字号
    plt.ylabel("loss", fontsize=14)  # Y轴标题及字号

    plt.show()

    print(len(Loss))

    print(Loss[100])




if __name__ =="__main__":

    logpath = '/media/wqg/3e165c12-9862-4867-b333-fbf93befd928/home/wqg/downloads/log/log/log.txt'
    show_log(logpath)







import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import numpy as np

if __name__=="__main__":
    number_list = []
    for i in range(0,1000000):
        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        while x**2 + y**2 > 1:
            x = random.uniform(-1,1)
            y = random.uniform(-1,1)
        number_list.append((x,y))
    histogram, xedge,yedge =np.histogram2d(x=[x[0] for x in number_list],y=[x[1] for x in number_list],density=True)
    fig,ax = plt.subplots(1,2)
    ax[0].hist([x[0] for x in number_list],bins=1000,density = True)
    conditional_list=[]
    for entry in number_list:
        if entry[1] > -0.05 and entry[1] <0.05:
            conditional_list.append(entry[0])
    ax[1].hist(conditional_list,bins=100,density=True)
    plt.show()
    threeD = plt.figure()
    axis = threeD.add_subplot(projection="3d")
    xpos,ypos = np.meshgrid(xedge[:-1],yedge[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    dx = (xedge[1] - xedge[0]) * 0.8
    dy = (yedge[1] - yedge[0]) * 0.8
    dz = histogram.ravel()
    axis.bar3d(xpos,ypos,zpos,dx,dy,dz)
    plt.show()
        
        
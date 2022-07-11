import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf
from matplotlib.patches import Rectangle
#from matplotlib.colors import ListedColormap, LinearSegmentedColormap

viridis = cm.get_cmap('viridis', 8)


    
def plot_DNN(V):
    nX = 111*2
    x = np.linspace(0,85,nX)
    y = np.linspace(-0.1,7.4,23)
    X,Y = np.meshgrid(x,y)
    X = X.flatten()
    Y = Y.flatten()
    
    nS = len(X)
    S = np.zeros([nS,5])
    S[:,3] = Y.transpose()
    S[:,4] = X.transpose()
    print(S[1,0:])
    
    Q = np.zeros([nS,9])
    
    idx = 0
    for s in S:
        state_tf = tf.convert_to_tensor(s)
        state_tf = tf.expand_dims(state_tf , 0)
        Q[idx,0:] = V(state_tf).numpy()
        idx+=1
    
    Values = Q[0:,4]
    Actions = np.argmax(Q,axis = 1)
    data = np.reshape(Values,[23,nX])
    data = np.transpose(data)
    ActionData = np.reshape(Actions,[23,nX])
    fig,ax = plt.subplots()
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    im = ax.imshow(data, cmap='YlGnBu',origin='lower',extent=[-0.1, 7.4,0, 85 ])
    ax.add_patch(Rectangle((0, 80), 4, 5,facecolor = 'red',edgecolor='red'))
    
    #ax.set_xlable([0,7.2])
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()
    
    #plt.scatter(S[4],S[3],c=Values)
    fig,ax = plt.subplots()
    ax.scatter(S[0:,3],Values)
    
    fig,ax = plt.subplots()
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    im = ax.imshow(ActionData, cmap='bone')
    
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()
    
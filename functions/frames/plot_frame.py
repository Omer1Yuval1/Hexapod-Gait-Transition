import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_frame(midline_0, midline_1, frame=-1, score=0, M1_0=None, M2_0=None, m1_0=None, m2_0=None, M1_1=None, M2_1=None, m1_1=None, m2_1=None, dv=None, lr=None):
    
    fig = plt.figure()
    
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
    
    spec = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[2, 1, 1])
    ax1 = fig.add_subplot(spec[0, 0], projection='3d')
    ax2 = fig.add_subplot(spec[0, 1], projection='3d')
    ax3 = fig.add_subplot(spec[1, 0])
    ax4 = fig.add_subplot(spec[1, 1])
    ax5 = fig.add_subplot(spec[2, :])
    
    # Plot A as a 3D line
    ax1.plot(midline_0[0], midline_0[1], midline_0[2], color='gray', label='Ref')
    ax2.plot(midline_1[0], midline_1[1], midline_1[2], color='gray', label='Sim')
    '''
    xlim_1 = ax1.get_xlim()
    xlim_1_diff = xlim_1[1] - xlim_1[0]
    d1 = xlim_1_diff * 0.025 # 0.1
    # d1 = 0.1
    
    xlim_2 = ax2.get_xlim()
    xlim_2_diff = xlim_2[1] - xlim_2[0]
    d2 = xlim_2_diff * 0.025
    # d2 = 0.1
    
    # Plot vectors from M at each corresponding point in A
    for i in range(len(m1_0)):
        ax1.quiver(midline_0[0, i], midline_0[1, i], midline_0[2, i], M1_0[0, i], M1_0[1, i], M1_0[2, i], color='red', length=d1, normalize=True)
        ax1.quiver(midline_0[0, i], midline_0[1, i], midline_0[2, i], M2_0[0, i], M2_0[1, i], M2_0[2, i], color='green', length=d1, normalize=True)
        ax2.quiver(midline_1[0, i], midline_1[1, i], midline_1[2, i], M1_1[0, i], M1_1[1, i], M1_1[2, i], color='red', length=d2, normalize=True)
        ax2.quiver(midline_1[0, i], midline_1[1, i], midline_1[2, i], M2_1[0, i], M2_1[1, i], M2_1[2, i], color='green', length=d2, normalize=True)
    '''
    
    ax1.set_aspect('equal', 'box')
    ax2.set_aspect('equal', 'box')
    
    ax1.legend()
    ax2.legend()
    
    ax1.set_title("Score=" + str(score))
    ax2.set_title("Frame=" + str(frame))
    
    if(not m1_0 is None):
        # print(np.mean(m1_0), np.mean(m1_1))
        ax3.plot(m1_0, label='m1')
        ax3.plot(m2_0, label='m2')
        ax3.plot(m1_1, label='m1_1')
        ax3.plot(m2_1, label='m1_2')
        ax4.plot(m1_1, label='m1')
        ax4.plot(m2_1, label='m2')
    
        ax3.legend()
        ax4.legend()
    
    ax5.plot(dv, label='dv')
    ax5.plot(lr, label='lr')
    
    if(frame >= 0):
        main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        plt.savefig(os.path.join(main_dir, 'resources', 'results', 'worm', 'ref_test', f"midline_{str(frame)}.png"), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
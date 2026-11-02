import numpy as np
import os
import matplotlib.pyplot as plt

def plot_swing_stance(contacts, save_dir, name, cmap=None, plot_flag=False):
    # print("contacts shape=", contacts.shape)
    # print(contacts)
    n_legs = 6
    # nt = (0.01 / P["dt_opt"]) # Scaling factor based on the time step.
    nh1 = 10 # Number of rows for each leg.
    nh2 = 5 # Number of rows between legs.

    if(cmap is None):
        cmap = 'gray'
    
    contact_mat = np.ones((n_legs*nh1 + (n_legs-1)*nh2, contacts.shape[0]))
    # contact_mat = np.zeros((n_legs*nh1 + (n_legs-1)*nh2, t.shape[0]))
    # print("contact_mat", contact_mat.shape)
    
    I = np.array([0,1,2,3,4,5])
    
    for i in range(I.shape[0]): # For each leg.
        y = contacts[:,I[i]] # Get contact forces for the current body part.
        # y = y[:int(t_max/dt)+1] # Trim [0,t_max].
        
        # spline = interpolate.UnivariateSpline(t, y, s=0.0001)
        # y = spline(y)
        
        # y = imbinarize(y, 'adaptive')
        ii = i*(nh1+nh2)
        contact_mat[ii:(ii+nh1),:] = np.tile(y,(nh1,1))
    
    # print(contact_mat.shape)
    
    if(plot_flag):
        fig2 = plt.figure() # figsize=(300,100)
        fig2.tight_layout()
        plt.gcf().set_size_inches(12, 3)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.imshow(contact_mat, cmap=cmap, vmin=0, vmax=1) # , vmin=0, vmax=50
        plt.xticks([])
        plt.yticks([])
        # plt.show()
        plt.savefig(os.path.join(save_dir,f"swing_stance_{name}.png"), bbox_inches='tight')

        plt.close(fig2)
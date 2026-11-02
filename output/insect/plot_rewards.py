import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(P, t, rewards, model_path, ep):

    # rewards

    # print(rewards["joint_phase_diffs"])

    # Plot weighted rewards
    fig, axs = plt.subplots(4, 2, figsize=(24, 16), sharex=True)

    axs[0,0].plot(t[1:], np.array(rewards["y_pos"]), label="progress")
    axs[0,0].plot(t[1:], np.array(rewards["side_motion"]), label="side_motion")
    axs[0,0].set_ylabel("progress")
    axs[0,0].legend()
    axs[0,0].grid()
    
    axs[1,0].plot(t[1:], np.array(rewards["z_pos"]), label="height")
    axs[1,0].set_ylabel("Height")
    axs[1,0].legend()
    axs[1,0].grid()

    if(P["gaits"][0]["type"] == "jump"):
        axs[2,0].plot(0, np.array([0]), label="contact_time")
    else:
        if(P["multi_gait"]):
            axs[2,0].plot(t[1:], np.array(rewards["contact_time"][:, 0]), '-b', lw=1, label="contact time reward (gait 1)")
            axs[2,0].plot(t[1:], np.array(rewards["contact_time"][:, 1]), '-r', lw=1, label="contact time reward (gait 2)")
            
            axs[3,0].plot(t[1:], np.array(rewards["intra_set_contact_std"][:, 0]), '-b', lw=0.5, label="intra_set_contact_std_0")
            axs[3,0].plot(t[1:], np.array(rewards["inter_set_contact_std"][:, 0]), '-r', lw=0.5, label="inter_set_contact_std_0")
            axs[3,0].plot(t[1:], np.array(rewards["intra_set_contact_std"][:, 1]), '--b', lw=0.5, label="intra_set_contact_std_1")
            axs[3,0].plot(t[1:], np.array(rewards["inter_set_contact_std"][:, 1]), '--r', lw=0.5, label="inter_set_contact_std_1")
        else:
            axs[2,0].plot(t[1:], np.array(rewards["contact_time"]), label="contact_time")
            axs[3,0].plot(t[1:], np.array(rewards["intra_set_contact_std"]), label="intra_set_contact_std")
            axs[3,0].plot(t[1:], np.array(rewards["inter_set_contact_std"]), label="inter_set_contact_std")
        # axs[2,0].plot(t[1:], np.array(rewards["contact_time"]) * P["gait_params"]["reward_weights"]["contact_time"], label="contact_time")
    
    axs[2,0].set_ylabel("Reward")
    axs[2,0].legend()
    axs[2,0].grid()

    axs[3,0].set_ylabel("Reward")
    axs[3,0].legend()
    axs[3,0].grid()
    

    axs[0,1].remove()
    axs[1,1].remove()
    ax01_polar = plt.subplot2grid((4, 2), (0, 1), rowspan=2, colspan=1, polar=True)
    ax01_polar.plot(np.array(rewards["gait_phase"]), t[1:], label="gait_phase")
    # ax2_twin = axs[2].twinx()
    if(P["gaits"][0]["type"] == "walk"):
        joint_phases = np.array(rewards["joint_phases"])
        ax01_polar.plot(joint_phases[:, 0], t[1:], label="joint_phases_L2")
        ax01_polar.plot(joint_phases[:, 1], t[1:], label="joint_phases_R2")
    # ax01_polar.set_ylabel("Reward")
    ax01_polar.legend()
    ax01_polar.grid()
    ax01_polar.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
    """ axs[2].plot(t[1:], np.array(rewards["angular_velocity"]) * P["gait_params"]["reward_weights"]["angular_velocity"], label="angular_velocity")
    ax01_polar.set_ylabel("Reward")
    ax01_polar.legend()
    ax01_polar.grid() """
    
    axs[2,1].remove()
    axs[3,1].remove()
    ax23_polar = plt.subplot2grid((4, 2), (2, 1), rowspan=2, colspan=1, polar=True)
    ax23_polar.plot(np.array(rewards["roll"]), t[1:], label="roll")
    ax23_polar.plot(np.array(rewards["pitch"]), t[1:], label="pitch")
    ax23_polar.plot(np.array(rewards["yaw"]), t[1:], label="yaw")
    # ax23_polar.set_xlabel("Time")
    # ax23_polar.set_ylabel("Angle [deg]")
    ax23_polar.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
    ax23_polar.grid()
    
    # plt.xlabel("Time")
    # plt.ylabel("Reward")
    # plt.legend()
    # plt.grid()
    plt.savefig(f"{model_path}/reward_{ep}.png")
    # plt.show()

    plt.close(fig)
freq = 1; % Hz.

if(freq == 1)
    SP_actions = 0.999; % [T=0.5: 0.9999, T=1: 0.999].
    source_dir = '165_1Hz';
elseif(freq == 2)
    SP_actions = 0.9999;
    source_dir = '167_2Hz';
end

LineWidth_1 = 2;
LineWidth_2 = 2;

frame_skip = 1;
dt = 0.01 * frame_skip;
plot_type = "qpos+actions"; % ["qpos_ref+qpos", "qpos+actions", "qpos+qvel"].
model_type = "locust"; % ["locust", "ant"].

close all;

rng(0);

if(model_type == "locust")
    filename_prefix = 'Locust_Walking_model=0_ep=0_0_';
    
    data_dir = ['D:\users\Omer\repos\insectRL\resources\output\results\six_legs\',source_dir,'\results'];
    
    if(0) % All legs.
        joint_names = ["L1_{J1}", "L1_{J2}", "R1_{J1}", "R1_{J2}", "L2_{J1}", "L2_{J2}", "R2_{J1}", "R2_{J2}", "L3_{J1}", "L3_{J2}", "R3_{J1}", "R3_{J2}"];
        joint_indices = [8,9,12,13,16,17,20,21,25,26,29,30];
        action_indices = [1,2,5,6,9,10,13,14,18,19,22,23];
    elseif(0) % Only front and middle legs.
        joint_names = ["L1_{J1}", "L1_{J2}", "R1_{J1}", "R1_{J2}", "L2_{J1}", "L2_{J2}", "R2_{J1}", "R2_{J2}"];
        joint_indices = [8,9,12,13,16,17,20,21];
        action_indices = [1,2,5,6,9,10,13,14];
    else
        joint_names = ["L3_{J1}", "L3_{J2}", "R3_{J1}", "R3_{J2}"];
        joint_indices = [25,26,29,30];
        action_indices = [18,19,22,23];

    end
elseif(model_type == "ant")
    
    filename_prefix = 'Ant_model=0_3_';
    data_dir = 'D:\users\Omer\repos\insectRL\resources\output\video_ant';
    
    joint_names = ["hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3", "hip_4", "ankle_4"];
    joint_indices = 8:15;
    action_indices = 1:8;
    
    % joint_names = ["hip_2", "hip_4"];
    % joint_indices = [10,14];
    % action_indices = [3,7];

    % joint_names = ["hip_2", "hip_3"];
    % joint_indices = [10,12];
    % action_indices = [3,5];
end

actions = readtable([data_dir,filesep,filename_prefix,'actions.csv']);
observations = readtable([data_dir,filesep,filename_prefix,'qpos.csv']);
cfrc_ext = readtable([data_dir,filesep,filename_prefix,'cfrc_ext.csv']);
contact_forces = readtable([data_dir,filesep,filename_prefix,'contact_forces.csv']);
leg_pos = readtable([data_dir,filesep,filename_prefix,'leg_pos.csv']);
% reference_observations = readtable([data_dir,filesep,filename_prefix,'qpos_ref.csv']);

if(plot_type == "qpos_ref+qpos")
    T = 3; % [0.25, 2].
elseif(plot_type== "qpos+actions")
    T = 4; % [2,4].
end

N = (T / dt);
disp(['Total number of time points: ', num2str(N)]);
t = 0:dt:(T-dt); % (T-2*dt); % ((size(actions,1)-1)*dt);
actions = actions(1:N,:);
observations = observations(1:N,:);
cfrc_ext = cfrc_ext(1:N,:);
contact_forces = contact_forces(1:N,:);
leg_pos = leg_pos(1:N,:);
% reference_observations = reference_observations(1:N,:);

CM = lines(6); % One color for each leg.
figure('WindowState','maximized');

% *****************************************************************************

if(plot_type == "qpos_ref+qpos")
    c = 0;
    ax1 = subplot(2,1,1); % Observation.
    hold(ax1,"on");
    for i=1:length(joint_indices)
        if(mod(i,2) == 1)
            c = c + 1;
            plot(t,reference_observations{:,joint_indices(i)} .* (180/pi) + 1*(rand(1)-0.5),'-','Color',CM(c,:),'LineWidth',LineWidth_1);
        else
            plot(t,reference_observations{:,joint_indices(i)} .* (180/pi) + 1*(rand(1)-0.5),'--','Color',CM(c,:),'LineWidth',LineWidth_1);
        end
    end
    title('Reference joint angles');
    % xlabel('Time [s]');
    ylabel('Joint angle [°]');
    set(ax1,'FontSize',22);
    grid(ax1,"on");
    legend(joint_names,'Location','eastoutside'); % ,'Orientation','horizontal'.
    set_subplot_position(ax1);
    % *****************************************************************************
end

if(plot_type == "qpos_ref+qpos") % Plot simulated joint angles.
    ax_qpos = 2;
else
    ax_qpos = 1;
end

c = 0;
ax2 = subplot(2,1,ax_qpos); % Observation.
plt = [];
m = table2array(mean(observations,[2,1]));
hold(ax2,"on");
for i=1:length(joint_indices)
    if(mod(i,2) == 1)
        c = c + 1;
        plt(end+1) = plot(t,(observations{:,joint_indices(i)}-m) .* (180/pi),'-','Color',CM(c,:),'LineWidth',LineWidth_2);
        % plot(t,observations{:,joint_indices(i)} .* (180/pi),'.k','MarkerSize',8);
    else
        plt(end+1) = plot(t,(observations{:,joint_indices(i)}-m) .* (180/pi),'--','Color',CM(c,:),'LineWidth',LineWidth_2);
    end
end
title('Joint angles'); % Simulated
% xlabel('Time [s]');
ylabel('Joint angle [°]');
set(ax2,'FontSize',22);
grid(ax2,"on");
legend(plt,joint_names,'Location','eastoutside'); % ,'Orientation','horizontal'.
set_subplot_position(ax2);
% ylim([-45,45]);
ylim([-45,45]+(m*(180/pi)));

% print('../resources/plots/ref_qpos+qpos.png','-dpng','-vector');

if(plot_type == "qpos+actions") % Plot joint torques.
    c = 0;
    ax2 = subplot(2,1,2); % Actions.
    plt = [];
    hold(ax2,"on");
    for i=1:length(action_indices)
        
        y = actions{:,action_indices(i)};
        fit_object = fit(t',y,'smoothingspline','smoothingparam',SP_actions);
        y = fit_object(t);

        if(mod(i,2) == 1)
            c = c + 1;
            plt(end+1) = plot(t,y,'-','Color',CM(c,:),'LineWidth',LineWidth_2);
            % plot(t,y,'.k','MarkerSize',8);
        else
            plt(end+1) = plot(t,y,'--','Color',CM(c,:),'LineWidth',LineWidth_2);
        end
    end
    title('Joint torques');
    xlabel('Time [s]');
    ylabel('Joint torque (normalized)');
    set(ax2,'FontSize',22);
    grid(ax2,"on");
    legend(plt,joint_names,'Location','eastoutside'); % ,'Orientation','horizontal'.
    set_subplot_position(ax2);
    ylim([-1.1,1.1]);

    % print('../resources/plots/qpos+actions.png','-dpng','-vector');
elseif(plot_type == "qpos+qvel") % Plot joint velocities.
    ax2 = subplot(2,1,2); % Actions.
    hold(ax2,"on");
    for i=1:length(action_indices)
        plot(t,actions{:,action_indices(i)},'LineWidth',LineWidth_1);
        plot(t,actions{:,action_indices(i)},'.k','MarkerSize',8);
    end
    title('Joint torques');
    xlabel('Time [s]');
    ylabel('Joint torque (normalized)');
    set(ax2,'FontSize',22);
    grid(ax2,"on");
    legend(joint_names,'Location','eastoutside'); % ,'Orientation','horizontal'.
    set_subplot_position(ax2);

    print('../resources/plots/qpos+actions.png','-dpng','-vector');
end

n_legs = 6;
nh1 = 10;
nh2 = 5;
contact_mat = zeros(n_legs*nh1 + (n_legs-2)*nh2,length(t));

I = 10:5:35;
figure;
for i=1:length(I) % size(contact_forces,2)
    y = contact_forces{:,I(i)};
    % fit_object = fit(t',y,'smoothingspline','smoothingparam',0.99);
    % y = fit_object(t);

    plot(y);
    hold on;
    % plot(y,'.k');
    
    % y = imbinarize(y, 'adaptive');

    ii = (i-1)*(nh1+nh2) + 1;
    contact_mat(ii:ii+(nh1-1),:) = repmat(y',nh1,1);
end
title("contact forces");

%{
thresh = 0.1;
contact_mat(contact_mat >= thresh) = 1;
contact_mat(contact_mat < thresh) = 0;
%}
contact_mat = 1 - contact_mat;
contact_mat(contact_mat == 0) = 0.3;
%}

figure('Position', [100, 100, size(contact_mat, 2), size(contact_mat, 1)]);
ax = axes('Parent', gcf, 'Position', [0, 0, 1, 1]);
imshow(contact_mat);
% print(['D:\users\Omer\repos\insectRL\resources\papers\plots\locust_stl\1Hz\swing_stance_chart_2Hz_T=',num2str(T),'s.svg'],'-dsvg','-vector');

%{
figure;
for i=1:length(I) % size(contact_forces,2)
    plot(cfrc_ext{:,I(i)});
    hold on;
    % plot(cfrc_ext{:,I(i)},'.k');
end
title("cfrc_ext");
%}

figure;
for i=(9+3+3):3:56
    y = leg_pos{:,i};
    fit_object = fit(t',y,'smoothingspline','smoothingparam',0.99);
    y = fit_object(t);
    
    y = y - mean(y);
    y = y / max(abs(y));
    y = y ./ 2;
    y = y - min(y);
    
    plot(t,y);

    hold on;

    findpeaks(y,t,'MinPeakProminence',0.1,'Annotate','extents');
    
end
title("Leg tip z-positions");

% x-y leg tip trajectories

thorax2head_angles = mod(atan2(leg_pos{:,2} - leg_pos{:,5},leg_pos{:,1} - leg_pos{:,4}),2*pi);
% Rz = rotz(-main_axis_angle * 180 / pi); % Rotation matrix to align the main body axis with [1,0,0].
% Rz = Rz(1:2,1:2);

% trajectories accordingly.
% figure;
% I = 10:6:45;
for i=10:6:45 % For each leg.
    x0 = leg_pos{:,i};
    y0 = leg_pos{:,i+1};
    x1 = leg_pos{:,i+3};
    y1 = leg_pos{:,i+4};

    base2tip = [y1 - y0 ; x1-x0];
    
    plot(x,y);
    
    hold on;
end
title("Leg tip trajectories");


function set_subplot_position(ax)
    p = get(ax,'Position');
    p(1) = 0.06;
    p(3) = 0.83;
    set(ax,'Position',p);
    
end

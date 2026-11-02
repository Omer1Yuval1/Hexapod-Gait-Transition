data_dir = 'D:\users\Omer\repos\insectRL\resources\output\csv';

data = dir([data_dir,filesep,'*.csv']);

dt = 0.01;
step_scale_factor = 10^6;
steps_moving_average_size = 100;

figure('WindowState','maximized');
ax1 = subplot(2,1,1);
ax2 = subplot(2,1,2);
for i=1:numel(data)
    if(contains(data(i).name, '(1)'))
        reward_data = readtable([data(i).folder,filesep,data(i).name]);
        hold(ax1,"on");
        plot(ax1, [reward_data.Step] ./ step_scale_factor, [reward_data.Value],'color',[0.2,0.2,0.2,0.2]);
        plot(ax1, [reward_data.Step] ./ step_scale_factor, smooth([reward_data.Value], steps_moving_average_size),'LineWidth',2);
        % reward_data(end+1).name = data(i).name;
        % reward_data(end).folder = data(i).folder;
        % reward_data(end).data = readtable([data(i).folder,filesep,data(i).name]);
    else
        episode_data = readtable([data(i).folder,filesep,data(i).name]);
        hold(ax2,"on");
        plot(ax2, [episode_data.Step] ./ step_scale_factor, [episode_data.Value].*dt, 'color',[0.2,0.2,0.2,0.2]);
        plot(ax2, [episode_data.Step] ./ step_scale_factor, smooth([episode_data.Value].*dt, steps_moving_average_size),'LineWidth',2);
        % episode_data(end+1).name = data(i).name;
        % episode_data(end).folder = data(i).folder;
        % episode_data(end).data = readtable([data(i).folder,filesep,data(i).name]);
    end
end

set(ax1, 'FontSize',22);
% xlabel(ax1, 'Step [per million]');
ylabel(ax1, 'Mean episode reward');
xlim(ax1, [0,50]);
grid(ax1,"on");
title(ax1,'Reward');

set(ax2, 'FontSize',22);
xlabel(ax2, 'Step [per million]');
ylabel(ax2, 'Mean episode length [s]');
xlim(ax2, [0,50]);
grid(ax2,"on");
title(ax2,'Mean episode length');

print('../resources/plots/reward+mean_ep_len.png','-dpng','-vector');
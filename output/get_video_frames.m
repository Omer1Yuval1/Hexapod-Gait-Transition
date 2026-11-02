source = 'D:\users\Omer\repos\insectRL\resources\output\results\six_legs\165_1Hz\results\Locust_Walking_model=0_ep=11_0_cam1-episode-0.mp4';
target = 'D:\users\Omer\repos\insectRL\resources\papers\plots\locust_stl\1Hz\frames\';

v=VideoReader(source);

frames = read(v, [1,inf]);

for i=1:10:100
    imwrite(frames(:,:,:,i),[target,num2str(i-1),'s.png'],'Mode','lossless');
end

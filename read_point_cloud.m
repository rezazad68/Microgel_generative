% cloud = read_point_cloud('LHA_FOCTS_NIPMAM_SINGLE.csv');
% pcshow(cloud)
%
function cloud = read_point_cloud(filename, apply_flip)

data  = dlmread(filename);

if(size(data,2)>3) 
    data = data(:,1:3);
end
if apply_flip
data(:,3) = 1-data(:,3); %flip z dimension
end
cloud = pointCloud(data);
pcshow(cloud); 
end















clear all
close all
clc

% Load configuration
config;

filtered_pointcloud = [];
%% Read sample pointcloud data
cloud = read_point_cloud(raw_pc_path, true);    

%% detect individual microgels:
[centers_x, centers_y, x_range, y_range] = detect_gels(cloud, 0);

%% Extract individual microgel and remove outlier 
for idx=1 : length(centers_x)
    gel1 = get_bounding_box(cloud, centers_x, centers_y, x_range, y_range, idx, 0, 0);
    filtered_point_cloud = removeOutliers(gel1.Location);
    filtered_pointcloud{idx} = filtered_point_cloud;
    
end

noise_removed = [];
for idx=1 : length(filtered_pointcloud)
noise_removed = [noise_removed; filtered_pointcloud{idx}];
end
ptCloud_unnoisy = pointCloud(noise_removed);

%% Calculate correlation matrix
[sorted_corr_sum, sorted_corr_sum2, sortIdx]=calculate_corrolation_index(filtered_pointcloud, use_spherical_information);

%% Elbow strategy
% Find the change in correlation values
sorted_corr_sum_norm = sorted_corr_sum/max(sorted_corr_sum);
sorted_corr_sum_norm2 = sorted_corr_sum2/max(sorted_corr_sum2);
delta_corr = diff(sorted_corr_sum_norm);
% Calculate the second derivative to find the inflection point
second_derivative = diff(delta_corr);
% Find the index of the point with the maximum second derivative
[max_second_derivative, idx] = max(second_derivative);
% The index 'idx' corresponds to the elbow point
threshold_index = idx;


%% Visualization of the results
selected_color = [0, 255, 0];
dropped_color  = [255, 0, 0];
draw_color     = [0, 0, 255];


N = fix(length(filtered_pointcloud)*Percentage_select);
M = fix(length(filtered_pointcloud)*(1-Percentage_drop)); 

pcaggregate = [];
color_aggregated = [];
for idx=1 : length(filtered_pointcloud)
pcaggregate = [pcaggregate; filtered_pointcloud{sortIdx(idx)}];
if idx<=N
rgbList = repmat(selected_color, length(filtered_pointcloud{sortIdx(idx)}), 1);
elseif idx<M && idx>N
rgbList = repmat(draw_color, length(filtered_pointcloud{sortIdx(idx)}), 1);
else
rgbList = repmat(dropped_color, length(filtered_pointcloud{sortIdx(idx)}), 1);
end

color_aggregated = [color_aggregated; rgbList];

end
threshold_index = N;
ptCloud = pointCloud(pcaggregate, 'Color', color_aggregated);

figure(1)
pcshow(cloud)
set(gcf,'color','w');
set(gca,'color','w');
set(gca,'XColor', 'k', 'YColor', 'k', 'ZColor', 'k');
title('Raw microgel point cloud data from SRFM measurements at temperatures 36 °C');

figure(2)
pcshow(ptCloud_unnoisy)
set(gcf,'color','w');
set(gca,'color','w');
set(gca,'XColor', 'k', 'YColor', 'k', 'ZColor', 'k');
title('Effect of removing noise');

figure(3)
pcshow(ptCloud)
set(gcf,'color','w');
set(gca,'color','w');
set(gca,'XColor', 'k', 'YColor', 'k', 'ZColor', 'k');
title('Clustering result');

figure(4)
% Create a colormap based on the length of the data
% cmap = [repmat([0, 1, 0], threshold_index, 1); repmat([1, 0, 0], length(sorted_corr_sum) - threshold_index, 1)];
cmap = [repmat([0,1,0],N,1); repmat([0,0,1],M-N-1,1); repmat([1,0,0], length(sorted_corr_sum)-M,1)]; % Green, Blue, Red

% Plot the scatter plot with the selected threshold
scatter(1:length(sorted_corr_sum_norm), sqrt(sorted_corr_sum_norm), 15, cmap, 'filled');
xlabel('Sample Index');
ylabel('Normalized Correlation Value');
title('Automatic sample selection results');

% Set custom x and y limits
xlim([0, length(sorted_corr_sum)]); % Customize x-axis limits
ylim([-0.2, 1.2]); % Customize y-axis limits

%% Visualize the selected samples
selected_list = {};
for idx=1 : N
selected_list{idx} =  filtered_pointcloud{sortIdx(idx)};
end
totals = length(selected_list);
numRows = fix(sqrt(totals));
numCols = fix(sqrt(totals))+1;
figure(5);

% Loop through each image and display it in a subplot
for i = 1:min(numRows*numCols, numel(selected_list))
    subplot(numRows, numCols, i);
    x = selected_list{i};
    rgbList = repmat(selected_color, length(x), 1);
    x = x - mean(x);
    x = pointCloud(x, 'Color', rgbList);
    pcshow(x);
%     set(gcf,'color','w');
%     set(gca,'color','w');
%     set(gca,'XColor', 'k', 'YColor', 'k', 'ZColor', 'k');
    zlim([-500 500])
    title(['Sample ', num2str(i)]);
    axis off;
end



%%
% Prompt the user for input
userInput = inputdlg('Enter the index of microgels to be dropped, separated by commas:', 'Input', [1 40]);

% Check if the user clicked cancel or entered an empty value
if isempty(userInput)
    disp('No input provided.');
    return;
end

% Split the input string into separate numbers using commas
inputNumbers = strsplit(userInput{1}, ',');

% Convert the input numbers to integers
integerList = str2double(inputNumbers);
if ~isnan(integerList)
selected_list(integerList) = [];
end
figure(6)
% Loop through each image and display it in a subplot
for i = 1:min(numRows*numCols, numel(selected_list))
    subplot(numRows, numCols, i);
    x = selected_list{i};
    rgbList = repmat(selected_color, length(x), 1);
    x = x - mean(x);
    x = pointCloud(x, 'Color', rgbList);
    pcshow(x);
%     set(gcf,'color','w');
%     set(gca,'color','w');
%     set(gca,'XColor', 'k', 'YColor', 'k', 'ZColor', 'k');
    zlim([-500 500])
    title(['Sample ', num2str(i)]);
    axis off;
end


if exist(save_path_selected, 'dir') == 0
    % Create the directory
    mkdir(save_path_selected);
end

for ids=1:numel(selected_list)
    pccloud = pointCloud(selected_list{ids});
    pcwrite(pccloud, strcat(save_path_selected, int2str(ids),'.ply'))
end

    
%% Convexhull fitting
figure(7)
Agg_selected = [];
% Loop through each image and display it in a subplot
for i = 1:min(numRows*numCols, numel(selected_list))
    x_agg = selected_list{i};
    x_agg = x_agg - mean(x_agg);
    Agg_selected = [Agg_selected; x_agg];
end
Agg_selected = removeOutliers(Agg_selected);
PC = Agg_selected;  % Replace with your point cloud data

% Define the selected color with transparency for the point cloud
selected_color = [0, 255, 128] / 255;
% selected_color = [0.5, 0.5, 0.5]; % Gray color with transparency, modify as needed

% Create a pointCloud object with the specified color for all points
rgbaList = repmat(selected_color, length(PC), 1);
ptCloud = pointCloud(PC, 'Color', rgbaList);

% Calculate the centroid of the point cloud
centroid = mean(PC);

% Calculate the squared distance from each point to the centroid
distancesSquared = sum((PC - centroid).^2, 2);

% Calculate the radius of the sphere as the square root of the maximum squared distance
radius = sqrt(max(distancesSquared));

% Fit a 3D convex hull to the point cloud
k = convhull(PC);

% Extract the vertices of the convex hull
hullVertices = PC(unique(k(:)), :);

% Create a figure for visualization

hold on;

% Plot the convex hull vertices
scatter3(hullVertices(:, 1), hullVertices(:, 2), hullVertices(:, 3), 30, [1, 0.647, 0], 'filled');

% Plot the point cloud with transparency
pcshow(ptCloud);

% Define two points for the pointer line: the centroid and a point above the centroid
pointer_start = centroid;
pointer_end = centroid + [0, 0, radius*1.2]; % Extend the line upward

% Plot the pointer line with increased line width
plot3([pointer_start(1), pointer_end(1)], ...
      [pointer_start(2), pointer_end(2)], ...
      [pointer_start(3), pointer_end(3)], 'r', 'LineWidth', 4); % Increased line width

% Display the radius value on top of the pointer
text(pointer_end(1), pointer_end(2), pointer_end(3), ...
     sprintf('Radius: %.2f', radius), 'Color', 'r', 'FontSize', 12, 'HorizontalAlignment', 'center');

xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
title('Convexhull of the Microgel with Radiou');

% Set axis limits if needed
maxRange = max(radius, max(sqrt(sum(PC.^2, 2))));
xlim([-maxRange, maxRange]);
ylim([-maxRange, maxRange]);
zlim([-maxRange, maxRange]);

hold off;
set(gcf,'color','w');
set(gca,'color','w');
set(gca,'XColor', 'k', 'YColor', 'k', 'ZColor', 'k');
title('Representaive sample for tempreture 36 °C');


function filteredPointCloud = removeOutliers(pointCloud)
    % Calculate the center of the point cloud
    center = mean(pointCloud);
    
    % Calculate the distances of each point to the center
    distances = sqrt(sum((pointCloud - center).^2, 2));
    
    % Sort the distances in descending order
    [~, sortedIndices] = sort(distances, 'descend');
    
    % Determine the number of points to remove (5% of the total points)
    numPointsToRemove = round(0.005 * size(pointCloud, 1));
    
    % Remove the points with the highest distances
    filteredPointCloud = pointCloud;
    filteredPointCloud(sortedIndices(1:numPointsToRemove), :) = [];
end










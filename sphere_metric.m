function[sphericalMetric]=sphere_metric(PC)
PC = removeOutliers(PC);
PC = PC - mean(PC); 
% min(PC) = -267.7670 -298.1811 -300.1648, max(PC) = 283.9330  293.8189  299.8352

% Define the selected color with transparency for the point cloud
selected_color = [0, 255, 128] / 255;
% selected_color = [0.5, 0.5, 0.5]; % Gray color with transparency, modify as needed

% Create a pointCloud object with the specified color for all points
rgbaList = repmat(selected_color, length(PC), 1);
ptCloud = pointCloud(PC, 'Color', rgbaList);

% Fit a 3D convex hull to the point cloud
k = convhull(PC);

% Extract the vertices of the convex hull
hullVertices = PC(unique(k(:)), :);

% Calculate the centroid of the point cloud
centroid = mean(PC);

% Calculate the distance from each convex hull vertex to the centroid
distancesToCentroid = sqrt(sum((hullVertices - centroid).^2, 2));

% Calculate the radius of the sphere as the average distance from convex hull vertices to the centroid
radius = mean(distancesToCentroid);


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

  
  
% Extract convex hull vertices in XY plane
hullXY = hullVertices(:, 1:2);

% Extract convex hull vertices in XZ plane
hullXZ = hullVertices(:, [1, 3]);

% Extract convex hull vertices in YZ plane
hullYZ = hullVertices(:, 2:3);

% Calculate diameters
diameterXY = mean( max(pdist2(hullXY, hullXY)));
diameterXZ = mean( max(pdist2(hullXZ, hullXZ)));
diameterYZ = mean( max(pdist2(hullYZ, hullYZ)));

% % Display results
% fprintf('Diameter in XY plane: %.2f units\n', diameterXY);
% fprintf('Diameter in XZ plane: %.2f units\n', diameterXZ);
% fprintf('Diameter in YZ plane: %.2f units\n', diameterYZ);


% Calculate the spherical metric
sphericalMetric = min([diameterXY, diameterXZ, diameterYZ]) / max([diameterXY, diameterXZ, diameterYZ]);

% % Display the spherical metric
% fprintf('Spherical Metric: %.2f\n', sphericalMetric);
end

function filteredPointCloud = removeOutliers(pointCloud)
    % Calculate the center of the point cloud
    center = mean(pointCloud);
    
    % Calculate the distances of each point to the center
    distances = sqrt(sum((pointCloud - center).^2, 2));
    
    % Sort the distances in descending order
    [~, sortedIndices] = sort(distances, 'descend');
    
    % Determine the number of points to remove (5% of the total points)
    numPointsToRemove = round(0.05 * size(pointCloud, 1));
    
    % Remove the points with the highest distances
    filteredPointCloud = pointCloud;
    filteredPointCloud(sortedIndices(1:numPointsToRemove), :) = [];
end



function [sorted_corr_sum,sorted_corr_sum_nosphe, sortIdx]=calculate_corrolation_index(gels, check_sym)
% Loop through each file in the list
density_vector = [];
for j = 1:length(gels)
    all_localiz = gels{j};
    zvec=all_localiz(:,3);
    xcoord=all_localiz(:,1);
    deltax = xcoord-median(xcoord);
    ycoord=all_localiz(:,2);
    deltay = ycoord-median(ycoord);
    % visualize one microgel in 3D space
    data = all_localiz;
    if(size(data,2)>3) 
        data = data(:,1:3);
    end
    cloud = pointCloud(data);
    % calculations and preparation of histogram ranges
    d_axis = sqrt(deltax.^2+deltay.^2);
    h = (zvec-median(zvec))+400;
    clear count V d_range h_range;


    d_dist=10;
    h_dist=10;

    labelx_dist=100;
    labely_dist=100;

    d_range=0:d_dist:400;
    h_range=0:h_dist:800;


    % calculation of the 2D histogram
    for i=1:length(d_range)-1
        V(i)=((d_range(i)+d_dist)^2-(d_range(i))^2)*pi*h_dist;
    end

    data=[d_axis h];
    for i=1:length(d_range)-1
        data((data(:,1)>d_range(i))&(data(:,1)<=d_range(i+1)),3)=i;
    end

    data2 = [d_axis h];
    for i=1:length(d_range)-1
        for j=1:length(data)
            if data2(j,1)>d_range(i)&&(data2(j,1)<=d_range(i+1))
             data2(j,3)=i;
            end
        end
    end    




    for i=1:length(h_range)-1
        data((data(:,2)>h_range(i))&(data(:,2)<=h_range(i+1)),4)=i;  
    end


    count=zeros(length(d_range)-1,length(h_range)-1);
    data=data(data(:,3)>0,:); % if a data point is out of the x range, throw it away
    data=data(data(:,4)>0,:);% if a data point is out of the y range, throw it away

    for i=1:size(data,1)
        try
            count(data(i,3),data(i,4))=count(data(i,3),data(i,4))+1; 

        catch
            i;
        end
    end


% visualization
figure_height_in_pixel=600;

figure;
set(gcf,'Position',[20 20 figure_height_in_pixel max(h_range)/max(d_range)*figure_height_in_pixel]+90);
count=count'./repmat(V,size(count,2),1);
density_vector = [density_vector reshape(count,[],1)];
imagesc(count); % To plot wrt density values

axis equal % suggestion by Eric to prevent elonagation in z



set(gca,'Ydir','Normal');
set(gca,'FontSize',30);
xlim([0.5 max(d_range/d_dist)+1]);
ylim([0.5 max(h_range/h_dist)+1]);

xlabel('Dist. from symmetry axis / nm','FontSize',30);
ylabel('Relative z / nm','FontSize',30);
set(gca,'XTick',d_range/d_dist*labelx_dist/d_dist+0.5);
set(gca,'XTickLabel',0:labelx_dist:max(d_range));
set(gca,'YTick',h_range/h_dist*labely_dist/h_dist+0.5);
set(gca,'YTickLabel',-400:labely_dist:400);
hcb = colorbar;

constant_colorbar = 1;
maximum_limcolor = 6e-03;
if constant_colorbar == 1 % keeping the colorbar constant
cmax = maximum_limcolor;
else 
cmax = max(max(count));
end
cmin=0;
caxis([cmin cmax]);
LowerIntensity = 0;
UpperIntensity = 1;
if LowerIntensity == 0 && UpperIntensity == 2000

  [map] = Colormapblue2red(colormap);% Use this for Radial solvato values
colormap(map); 

elseif LowerIntensity == 0 && UpperIntensity == 200
    colormap gray;


else
[map] = ColormapAll(colormap);
colormap(map); 

end



%save_fd = 'C:\Users\Berk Alperen Bener\Desktop\Arbeiten\Hiwi\Microgel-master\4_Microgel_plotter_V1_density\PAINT_DiffTemp\Core-shell\';
%fname_2d = strcat(save_fd,folder_name_extended,"\",file_name,'_2d-density-distr.png');
%saveas(gcf,fname_2d)

end
% end Ashvini code ends


%
if check_sym
symmetry = [];
for j=1:length(gels)
% Sample point cloud data (replace this with your actual data)
pc = gels{j};

% Extract x, y, and z coordinates
x = pc(:, 1);
y = pc(:, 2);
z = pc(:, 3);

% Compute skewness for each dimension
skewnessX = skewness(x);
skewnessY = skewness(y);
skewnessZ = skewness(z);

% Calculate symmetry score between 0 and 1 (lower value indicates less symmetry)
symmetryScoreX = 1 - abs(skewnessX);
symmetryScoreY = 1 - abs(skewnessY);
symmetryScoreZ = 1 - abs(skewnessZ);
a = [symmetryScoreX symmetryScoreY symmetryScoreZ];

symmetry = [symmetry; a];
end
end
%     %% Calculate the correlation
close all;

sphericial_metrics = [];
for j = 1:length(gels)
    all_localiz = gels{j};
    a = sphere_metric(all_localiz);
    sphericial_metrics = [sphericial_metrics; a];
    
end
    
if check_sym
index_max=0;
alpha = 0.7;
corr_mat = zeros(size(density_vector, 2),size(density_vector, 2));
symm_mat = zeros(size(density_vector, 2),size(density_vector, 2));
for cor_i=1:size(density_vector, 2)
    sum_cur=0;
    for cor_j=1:size(density_vector, 2)
        r = corrcoef(density_vector(:,cor_i), density_vector(:,cor_j));
        r2 = abs(1- abs(pdist([symmetry(cor_i, :); symmetry(cor_j, :)])));
        corr_mat(cor_i,cor_j)=r(1,2);
        symm_mat(cor_i,cor_j)=r2;
    end
end    
else
index_max=0;
corr_mat = zeros(size(density_vector, 2),size(density_vector, 2));

for cor_i=1:size(density_vector, 2)
    sum_cur=0;
    for cor_j=1:size(density_vector, 2)
        r = corrcoef(density_vector(:,cor_i), density_vector(:,cor_j));
        corr_mat(cor_i,cor_j)=r(1,2);
    end
end
end

% sort A in descending order (decreasing A values) 
% and keep the sort index in "sortIdx"
summat = sum(corr_mat);
temp = summat;
if check_sym
symmat = sum(symm_mat);
for i=1:size(density_vector, 2)
   
    if symmat(i)<mean(symmat)
        summat(i) = summat(i)-((1-sphericial_metrics(i))*100);
    end
    
end
end
[sorted_corr_sum_nosphe,tempp] = sort(temp,'descend');
[sorted_corr_sum,sortIdx] = sort(summat,'descend');

close all;












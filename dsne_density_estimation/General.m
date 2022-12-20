%% Initial setup
close all; clear; clc

% Loading the data (copied the mat file from sklearn)
mnist = load('mldata/mnist-original.mat');
X = single(mnist.data');
y = mnist.label';

%% Randomly select cluster of images at each site
% Things to try
% a) Randomize images at each local site
% b) Dirichlet mixtures instead of gaussian mixture models
% c) unequal cluster sizes at each local site
% 
site1_images = randperm(10, 9) - 1;
site2_images = randperm(9, 3) - 1;
site3_images = randperm(9, 3) - 1;

%% Extract images to local site 01
site1_Xa = X(y == site1_images(1), :);
site1_Xb = X(y == site1_images(2), :);
site1_Xc = X(y == site1_images(3), :);
site1_Xd = X(y == site1_images(4), :);
site1_Xe = X(y == site1_images(5), :);
site1_Xf = X(y == site1_images(6), :);
site1_Xg = X(y == site1_images(7), :);
site1_Xh = X(y == site1_images(8), :);
site1_Xi = X(y == site1_images(9), :);

site1_X = [site1_Xa(1:500, :); site1_Xb(1:500, :); site1_Xc(1:500, :); ...
            site1_Xd(1:500, :); site1_Xe(1:500, :); site1_Xf(1:500, :); ...
            site1_Xg(1:500, :); site1_Xh(1:500, :); site1_Xi(1:500, :)];
        
r = repmat(site1_images', 1, 500)';
site1_y = r(:);

% site1_X = X(ismember(y, site1_images), :);
% site2_X = X(ismember(y, site2_images), :);
% site3_X = X(ismember(y, site3_images), :);

% site1_y = ismember(y. site1_images)
% site2_y = ismember(y. site2_images)
% site3_y = ismember(y. site3_images)

%% ### Extract images to local site 02
% site2_X = X(y == site2_images(1) | ...
%             y == site2_images(2) | ...
%             y == site2_images(3) , :);
% 
% site2_y = y(y == site2_images(1) | ...
%             y == site2_images(2) | ...
%             y == site2_images(3));
        
site2_Xa = X(y == site2_images(1), :);
site2_Xb = X(y == site2_images(2), :);
site2_Xc = X(y == site2_images(3), :);

site2_X = [site2_Xa(1:500, :); site2_Xb(1:500, :); site2_Xc(1:500, :)];
r = repmat(site2_images', 1, 500)';
site2_y = r(:);

%% ### Extract images to local site 03
% site3_X = X(y == site3_images(1) | ...
%             y == site3_images(2) | ...
%             y == site3_images(3) , :);
% 
% site3_y = y(y == site3_images(1) | ...
%             y == site3_images(2) | ...
%             y == site3_images(3));

site3_Xa = X(y == site3_images(1), :);
site3_Xb = X(y == site3_images(2), :);
site3_Xc = X(y == site3_images(3), :);

site3_X = [site3_Xa(1:500, :); site3_Xb(1:500, :); site3_Xc(1:500, :)];
r = repmat(site3_images', 1, 500)';
site3_y = r(:);

%% Time for density estimation (at each local site)
% fit a Gaussian Mixture Model with 3 components at each site
options = statset('Display','final');
obj1 = fitgmdist(site1_X, 9, 'RegularizationValue', 1, 'Options', options);
obj2 = fitgmdist(site2_X, 3, 'RegularizationValue', 1, 'Options', options);
obj3 = fitgmdist(site3_X, 3, 'RegularizationValue', 1, 'Options', options);

%% Density aggregation at the remote site
mu1 = obj1.mu;
sigma1 = obj1.Sigma;
w1 = obj1.ComponentProportion;

mu2 = obj2.mu;
sigma2 = obj2.Sigma;
w2 = obj2.ComponentProportion;

mu3 = obj3.mu;
sigma3 = obj3.Sigma;
w3 = obj3.ComponentProportion;

mu = [mu1; mu2; mu3];
sigma = cat(3, sigma1, sigma2, sigma3);
p = [w1 w2 w3] / 3;

obj = gmdistribution(mu1, sigma1, w1);

%% Resampling at each of the local sites (entails sending the distribution to the local sites)
sampled_data1 = random(obj, 1000);
sampled_data2 = random(obj, 1000);
sampled_data3 = random(obj, 1000);

%% performing tSNE at each of the local sites and send the data to the remote site
augX_1 = [site1_X; sampled_data1];
augX_2 = [site2_X; sampled_data2];
augX_3 = [site3_X; sampled_data3];

augY_1 = [site1_y; NaN(1000, 1)];
augY_2 = [site2_y; NaN(1000, 1)];
augY_3 = [site3_y; NaN(1000, 1)];

%% Perform tsne at each of the local sites
figure
y_1 = tsne(augX_1, augY_1);
close

figure
y_2 = tsne(augX_2, augY_2);
close

figure
y_3 = tsne(augX_3, augY_3);
close
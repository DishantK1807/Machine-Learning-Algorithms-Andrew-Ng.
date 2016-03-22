% Loading image
A = double(imread('//home/jigyasa/Desktop/Machine-Learning-Algorithms/K-Means/cham.jpg'));

% Pixels in picture's length/width
dim = size(A,1);

% Number of Colors to represent
k = 16; 

% Initialize means to randomly-selected colors in the original photo.
means = zeros(k, 3);
rand_x = ceil(dim*rand(k, 1));
rand_y = ceil(dim*rand(k, 1));

for i = 1:k
    means(i,:) = A(rand_x(i), rand_y(i), :);
end


% Array to store the nearest neighbor for every pixel in the image
nearest_mean = zeros(dim);

% Run k-means
max_iterations = 100;
for itr = 1:max_iterations
    
    % SMeans to be calculated in this iteration
    new_means = zeros(size(means));
    
    % num_assigned(n) stores the number of pixels clustered around the nth mean
    num_assigned = zeros(k, 1);
    
    % For every pixel in the image, calculate the nearest mean and updating the means.
    for i = 1:dim
    
        for j = 1:dim
            % Calculating the nearest mean for the pixels in the image
            r = A(i,j,1); g = A(i,j,2); b = A(i,j,3);
            diff = ones(k,1)*[r, g, b] - means;
            distance = sum(diff.^2, 2);
            [val ind] = min(distance);
            nearest_mean(i,j) = ind;
            
            % Adding this pixel to the rgb values of its nearest mean
            new_means(ind, 1) = new_means(ind, 1) + r;
            new_means(ind, 2) = new_means(ind, 2) + g;
            new_means(ind, 3) = new_means(ind, 3) + b;
            num_assigned(ind) = num_assigned(ind) + 1;
        end
        
    end
    
    % Calculating new means
    for i = 1:k
        % Only update the mean if there are pixels assigned to it
        if (num_assigned(i) > 0)
            new_means(i,:) = new_means(i,:) ./ num_assigned(i);
        end
    end
    
    % Convergence test. Display by how much the means values are changing
    d = sum(sqrt(sum((new_means - means).^2, 2)))
    if d < 1e-5
        break
    end
    
    means = new_means;
end
disp(itr)

means = round(means);

% Recalculating the (big) image and display
large_image = double(imread('/home/jigyasa/Desktop/Machine-Learning-Algorithms/K-Means/cham.jpg'));
large_dim = size(large_image, 1);

for i = 1:large_dim
    for j = 1:large_dim
        r = large_image(i,j,1); g = large_image(i,j,2); b = large_image(i,j,3);
        diff = ones(k,1)*[r, g, b] - means;
        distance = sum(diff.^2, 2);
        [val ind] = min(distance);
        large_image(i,j,:) = means(ind,:);
    end 
end
imshow(uint8(round(large_image))); hold off

% Saving image
imwrite(uint8(round(large_image)), '/home/jigyasa/Desktop/Machine-Learning-Algorithms/K-Means/cham_kmeans.jpg');


% Displaying the mean colors (Matlab Only; Unfortunately, the rectangle function does not work in Octave)
% figure; hold on
% for i=1:k
%    col = (1/255).*means(i,:);
%    rectangle('Position', [i, 0, 1, 1], 'FaceColor', col, 'EdgeColor', col);
% end
% axis off

clc; clear all;

%rmdir('Aug', 's')
mkdir Aug
mkdir Aug/images
mkdir Aug/annotations

input_width = 400;
input_height = 192;

images = imageDatastore('ToTrain/images/', 'IncludeSubfolders', true);
groundTruth = imageDatastore('ToTrain/annotations/', 'IncludeSubfolders', true);

assert(length(images.Files) == length(groundTruth.Files));

for currImage = 1 : length(images.Files)
    [image, imageSegInfo] = readimage(images, currImage);
    [filepath,img_name,ext] = fileparts(imageSegInfo.Filename);

    [ground, groundSegInfo] = readimage(groundTruth, currImage);
    [groundFilepath,ground_name,groundExt] = fileparts(imageSegInfo.Filename);

    assert(strcmp(img_name, ground_name))

    imwrite(image, strcat("./Aug/images/", img_name, ".tif"))
    imwrite(ground, strcat("./Aug/annotations/", img_name, ".tif"))

    fprintf(1, '\t\tNow reading %d of %d\n', currImage, length(images.Files));
    original_center_img = center_crop(image, input_width, input_height, 0, 0);
    original_center_ground = center_crop(ground, input_width, input_height, 0, 0);

    imwrite(original_center_img, strcat("./Aug/images/center_", img_name, ".tif"))
    imwrite(original_center_ground, strcat("./Aug/annotations/center_", img_name, ".tif"))

    img_noise = imnoise(original_center_img,'gaussian',0.04);
    % ground_noise = imnoise(original_center_ground,'gaussian',0.04);

    imwrite(img_noise, strcat("./Aug/images/gaussian_", img_name, ".tif"))
    imwrite(original_center_ground, strcat("./Aug/annotations/gaussian_", img_name, ".tif"))

    img_noise_salt_and_pepper = imnoise(original_center_img,'salt & pepper',0.04);
    % ground_noise_salt_and_pepper = imnoise(original_center_img,'salt & pepper',0.04);

    imwrite(img_noise_salt_and_pepper, strcat("./Aug/images/salt_and_pepper_", img_name, ".tif"))
    imwrite(original_center_ground, strcat("./Aug/annotations/salt_and_pepper_", img_name, ".tif"))

    img_resized = imresize(image, [input_width input_height]);
    ground_resized = imresize(ground, [input_width input_height]);

    imwrite(img_resized, strcat("./Aug/images/resized_", img_name, ".tif"))
    imwrite(ground_resized, strcat("./Aug/annotations/resized_", img_name, ".tif"))

    img_rotated = imrotate(original_center_img, 5);
    ground_rotated = imrotate(original_center_ground, 5);

    imwrite(img_rotated, strcat("./Aug/images/rotated_", img_name, ".tif"))
    imwrite(ground_rotated, strcat("./Aug/annotations/rotated_", img_name, ".tif"))
end

function [center_img] = center_crop(img, width, height, shiftwidth, shiftheight )
    img_size = size(img);
    heightbot = fix( (img_size(1)-height)/2 + shiftheight);
    heighttop = heightbot+height;
    widthleft = fix((img_size(2)-width)/2 + shiftwidth);
    widthright = widthleft+width;
    center_img = img(heightbot:heighttop, widthleft:widthright,1:3);
end
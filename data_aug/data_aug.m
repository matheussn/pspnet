clc; clear all;

rmdir('Aug', 's')
mkdir Aug
mkdir Aug/images
mkdir Aug/annotations
mkdir Aug/images/healthy
mkdir Aug/annotations/healthy

input_width = 400;
input_height = 192;
target = ["healthy"];
shifting_value = [100,    0, -100,    0;
                    0   100,    0, -100];

for class = 1 : length(target)
    fprintf('\t%s\n', target(class));
    images = imageDatastore(strcat('dataset/', target(class)), 'IncludeSubfolders', true);
    groundTruth = imageDatastore(strcat('ground/', target(class)), 'IncludeSubfolders', true);

    assert(length(images.Files) == length(groundTruth.Files));

    for currImage = 1 : length(images.Files)
        [image, imageSegInfo] = readimage(images, currImage);
        [filepath,img_name,ext] = fileparts(imageSegInfo.Filename);

        [ground, groundSegInfo] = readimage(groundTruth, currImage);
        [groundFilepath,ground_name,groundExt] = fileparts(imageSegInfo.Filename);

        assert(strcmp(img_name, ground_name))

        imwrite(image, strcat("./Aug/images/", target(class), "/", img_name, ".tif"))
        imwrite(ground, strcat("./Aug/annotations/", target(class), "/", img_name, ".tif"))

        fprintf(1, '\t\tNow reading %d of %d\n', currImage, length(images.Files));
        original_center_img = center_crop(image, input_width, input_height, 0, 0);
        original_center_ground = center_crop(ground, input_width, input_height, 0, 0);

        imwrite(original_center_img, strcat("./Aug/images/", target(class), "/", img_name, "_center.tif"))
        imwrite(original_center_ground, strcat("./Aug/annotations/", target(class), "/", img_name, "_center.tif"))

        img_noise = imnoise(original_center_img,'gaussian',0.04);
        % ground_noise = imnoise(original_center_ground,'gaussian',0.04);

        imwrite(img_noise, strcat("./Aug/images/", target(class), "/", img_name, "_gaussian.tif"))
        imwrite(original_center_ground, strcat("./Aug/annotations/", target(class), "/", img_name, "_gaussian.tif"))

        img_noise_salt_and_pepper = imnoise(original_center_img,'salt & pepper',0.04);
        % ground_noise_salt_and_pepper = imnoise(original_center_img,'salt & pepper',0.04);

        imwrite(img_noise_salt_and_pepper, strcat("./Aug/images/", target(class), "/", img_name, "_salt_and_pepper.tif"))
        imwrite(original_center_ground, strcat("./Aug/annotations/", target(class), "/", img_name, "_salt_and_pepper.tif"))

        img_resized = imresize(image, [input_width input_height]);
        ground_resized = imresize(ground, [input_width input_height]);

        imwrite(img_resized, strcat("./Aug/images/", target(class), "/", img_name, "_resized.tif"))
        imwrite(ground_resized, strcat("./Aug/annotations/", target(class), "/", img_name, "_resized.tif"))

        img_rotated = imrotate(original_center_img, 5);
        ground_rotated = imrotate(original_center_ground, 5);

        imwrite(img_rotated, strcat("./Aug/images/", target(class), "/", img_name, "_rotated.tif"))
        imwrite(ground_rotated, strcat("./Aug/annotations/", target(class), "/", img_name, "_rotated.tif"))

        for i=1:4
          center_img = center_crop(image, input_width, input_height, shifting_value(1,i), shifting_value(2,i));
          center_ground = center_crop(ground, input_width, input_height, shifting_value(1,i), shifting_value(2,1));

          imwrite(center_img, strcat("./Aug/images/", target(class), "/", img_name, "_randon_", i, ".tif"));
          imwrite(center_ground, strcat("./Aug/annotations/", target(class), "/", img_name, "_randon_", i, ".tif"));
        end
    end
end

function [center_img] = center_crop(img, width, height, shiftwidth, shiftheight )
    img_size = size(img);
    heightbot = fix( (img_size(1)-height)/2 + shiftheight);
    heighttop = heightbot+height;
    widthleft = fix((img_size(2)-width)/2 + shiftwidth);
    widthright = widthleft+width;
    center_img = img(heightbot:heighttop, widthleft:widthright,1:3);
end


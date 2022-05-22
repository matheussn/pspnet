clc; clear all;

data_root = 'Aug_NRR';

rmdir('Aug_NRR', 's')
mkdir Aug_NRR
mkdir Aug_NRR/images
mkdir Aug_NRR/annotations

input_width = 300;
input_height = 192;

images = imageDatastore('ToTrain/images/', 'IncludeSubfolders', true);
groundTruth = imageDatastore('ToTrain/annotations/', 'IncludeSubfolders', true);

assert(length(images.Files) == length(groundTruth.Files));

to_filter = [];
to_filter_names = [];

for currImage = 1 : length(images.Files)
    [image, imageSegInfo] = readimage(images, currImage);
    [filepath,img_name,ext] = fileparts(imageSegInfo.Filename);

    [ground, groundSegInfo] = readimage(groundTruth, currImage);
    [groundFilepath,ground_name,groundExt] = fileparts(imageSegInfo.Filename);

    assert(strcmp(img_name, ground_name))

    copyfile(strcat("ToTrain/images/", img_name, ".tif"), strcat("./", data_root, "/images/", img_name, ".tif"));
    copyfile(strcat("ToTrain/annotations/", img_name, ".tif"), strcat("./", data_root, "/annotations/", img_name, ".tif"));

    fprintf(1, '\t\tNow reading %d of %d\n', currImage, length(images.Files));

    image = image(:, :,1:3);
    apply_filters(image, ground, img_name, data_root)

    rotation = randi([-20,20],1,1);
    img_rotated = imrotate(image, rotation);
    ground_rotated = imrotate(ground, rotation);

    rotated_img_name = strcat(img_name, "_rotated.tif");
    imwrite(img_rotated, strcat("./", data_root, "/images/", rotated_img_name))
    imwrite(ground_rotated, strcat("./", data_root, "/annotations/", rotated_img_name))
    apply_filters(img_rotated, ground_rotated, rotated_img_name, data_root)
end

function [] = apply_filters(img, ground, name, data_root)
    img_noise = imnoise(img,'gaussian',0.04);

    imwrite(img_noise, strcat("./", data_root, "/images/", name, "_gaussian.tif"))
    imwrite(ground, strcat("./", data_root, "/annotations/", name, "_gaussian.tif"))

    img_noise_salt_and_pepper = imnoise(img,'salt & pepper',0.04);

    imwrite(img_noise_salt_and_pepper, strcat("./", data_root, "/images/", name, "_salt_and_pepper.tif"))
    imwrite(ground, strcat("./", data_root, "/annotations/", name, "_salt_and_pepper.tif"))
end

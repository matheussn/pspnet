clc; clear all;

data_root = 'Aug_noises';

rmdir('Aug_noises', 's')
mkdir Aug_noises
mkdir Aug_noises/images
mkdir Aug_noises/annotations

input_width = 300;
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

    copyfile(strcat("ToTrain/images/", img_name, ".tif"), strcat("./", data_root, "/images/", img_name, ".tif"));
    copyfile(strcat("ToTrain/annotations/", img_name, ".tif"), strcat("./", data_root, "/annotations/", img_name, ".tif"));

    fprintf(1, '\t\tNow reading %d of %d\n', currImage, length(images.Files));
    image = image(:, :,1:3);
    img_noise = imnoise(image,'gaussian',0.04);

    imwrite(img_noise, strcat("./", data_root, "/images/", img_name, "_gaussian.tif"))
    imwrite(ground, strcat("./", data_root, "/annotations/", img_name, "_gaussian.tif"))

    img_noise_salt_and_pepper = imnoise(image,'salt & pepper',0.04);

    imwrite(img_noise_salt_and_pepper, strcat("./", data_root, "/images/", img_name, "_salt_and_pepper.tif"))
    imwrite(ground, strcat("./", data_root, "/annotations/", img_name, "_salt_and_pepper.tif"))
end

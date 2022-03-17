clc; clear all;

dsType = [
    "pspnet_100",
    "pspnet_100_adam",
    "pspnet_150",
    "pspnet_150_adam",
    "pspnet_50",
    "pspnet_50_adam",
    "psp_resnest_100_adam",
    "psp_resnest_50",
    "psp_unet_100",
    "psp_unet_100_adam",
    "psp_unet_150",
    "psp_unet_150_adam",
    "psp_unet_50",
    "psp_unet_50_adam"
    ];
target = ["healthy", "mild", "moderate", "severe"]; % Chose a folder containing images of one class.

for currDS = 1 : length(dsType) % iterating over different result folders
    fprintf('Init in folder %s \n', dsType(currDS));
    for currTG = 1 : length(target) % iterating over the lesions classes
        fprintf('\t%s\n', target(currTG));
        path = fullfile(pwd, 'work_dirs/', dsType(currDS), '/raw_res/', target(currTG));
        maskImds = imageDatastore(path, 'IncludeSubfolders', true);
        for currImage = 1 : length(maskImds.Files)
            [maskIm, segInfo] = readimage(maskImds, currImage); % reads one result image
            [filepath,name,ext] = fileparts(segInfo.Filename);

            orName = convertStringsToChars(strcat('dataset/', target(currTG), '/', name, '.tif')); % reads the groudtruth image with the same name as the mask
            orIm = imread(orName);

            fprintf(1, '\t\tNow reading %d of %d\n', currImage, length(maskImds.Files));

            finalMask = cat(3, maskIm(:,:,1), maskIm(:,:,1), maskIm(:,:,1)); % Creates a 3-dimensional image for the post processing step
            %finalMask = imcrop(maskIm, [74,174,449,249]); % cropping borders from the CNN
            finalIm(:,:,1) = orIm(:,:,1);
            finalIm(:,:,2) = orIm(:,:,2);
            finalIm(:,:,3) = orIm(:,:,3);
            finalIm(finalMask<200) = 0;

            [finalMaskPost, finalImPost] = postProc(finalMask, orIm);

            %%%%% SAVING FILES WITHOUT POST-PROCESSING %%%%%
            pathFinalMaskName = convertStringsToChars(strcat('segmentation_results/', dsType(currDS), '/pre/masks/', target(currTG), '/'));
            pathFinalImName = convertStringsToChars(strcat('segmentation_results/', dsType(currDS), '/pre/segmentation/', target(currTG), '/'));
            if ~exist(pathFinalMaskName, 'dir')
                mkdir(pathFinalMaskName);
                mkdir(pathFinalImName);
            end

            pathFinalMaskName = strcat(pathFinalMaskName, '/', name, ext);
            pathFinalImName = strcat(pathFinalImName, '/', name, ext);

            imwrite(finalMask, pathFinalMaskName);
            imwrite(finalIm, pathFinalImName);

            %%%%% SAVING POST-PROCESSED FILES %%%%%
            pathFinalMaskPostName = convertStringsToChars(strcat('segmentation_results/', dsType(currDS), '/post/masks/', target(currTG), '/'));
            pathFinalImPostName = convertStringsToChars(strcat('segmentation_results/', dsType(currDS), '/post/segmentation/', target(currTG), '/'));
            if ~exist(pathFinalMaskPostName, 'dir')
                mkdir(pathFinalMaskPostName);
                mkdir(pathFinalImPostName);
            end

            pathFinalMaskPostName = strcat(pathFinalMaskPostName, '/', name, ext);
            pathFinalImPostName = strcat(pathFinalImPostName, '/', name, ext);

            imwrite(finalMaskPost, pathFinalMaskPostName);
            imwrite(finalImPost, pathFinalImPostName);
        end
    end
end
% fprintf(1, 'XXXXXXXXXXX ENDCODE XXXXXXXXXXXX\n');
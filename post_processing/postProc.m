function [finalMask, finalIm5, finalIm4, finalIm3, finalIm2, finalIm1] = postProc(inputImage, orIm)
    finalIm1(:,:,1) = orIm(:,:,1);
    finalIm1(:,:,2) = orIm(:,:,2);
    finalIm1(:,:,3) = orIm(:,:,3);
    finalIm1(inputImage<200) = 0;

    binaryIm = imbinarize(rgb2gray(inputImage));
    se = strel('diamond',1);

    dilatedIm = imdilate(binaryIm, se); % Dilation operation
    finalMask = im2uint8(dilatedIm);
    finalMask = cat(3, finalMask, finalMask, finalMask);
    firstMask = finalMask;
    finalIm2(:,:,1) = orIm(:,:,1);
    finalIm2(:,:,2) = orIm(:,:,2);
    finalIm2(:,:,3) = orIm(:,:,3);
    finalIm2(finalMask<200) = 0;

    filledIm = imfill(dilatedIm,'holes'); % Hole-filling operation
    finalMask = im2uint8(filledIm);
    finalMask = cat(3, finalMask, finalMask, finalMask);
    secondMask = finalMask;
    finalIm3(:,:,1) = orIm(:,:,1);
    finalIm3(:,:,2) = orIm(:,:,2);
    finalIm3(:,:,3) = orIm(:,:,3);
    finalIm3(finalMask<200) = 0;

    erodedIm = imerode(filledIm, se); % Erosion operation

    finalMask = im2uint8(erodedIm);
    finalMask = cat(3, finalMask, finalMask, finalMask);
    thirdMask = finalMask;
    finalIm4(:,:,1) = orIm(:,:,1);
    finalIm4(:,:,2) = orIm(:,:,2);
    finalIm4(:,:,3) = orIm(:,:,3);
    finalIm4(finalMask<200) = 0;

    smallRemovedIm = bwareaopen(erodedIm, 30); % Remove objetcs smaller than 30 pixels
    finalMask = cat(3, smallRemovedIm, smallRemovedIm, smallRemovedIm);

    finalMask = im2uint8(finalMask);
    fourMask = finalMask;
    finalIm5(:,:,1) = orIm(:,:,1);
    finalIm5(:,:,2) = orIm(:,:,2);
    finalIm5(:,:,3) = orIm(:,:,3);
    finalIm5(finalMask<200) = 0;

    dilatedIm = cat(3, dilatedIm, dilatedIm, dilatedIm);
    dilatedIm = im2uint8(dilatedIm);

    filledIm = cat(3, filledIm, filledIm, filledIm);
    filledIm = im2uint8(filledIm);

    erodedIm = cat(3, erodedIm, erodedIm, erodedIm);
    erodedIm = im2uint8(erodedIm);
end
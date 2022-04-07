clc; clear all;

[maskIm, segInfo] = readimage('image003-2-roi2.tif');
finalMask = cat(3, maskIm(:,:,1), maskIm(:,:,1), maskIm(:,:,1));
finalIm(:,:,1) = orIm(:,:,1);
finalIm(:,:,2) = orIm(:,:,2);
finalIm(:,:,3) = orIm(:,:,3);
finalIm(finalMask<200) = 0;

[finalMaskPost, finalImPost] = postProc(finalMask, orIm);
clc; clear all;
dsType = ["nopreproc_nadam", "nopreproc_nadamPost"];
target = ["healthy", "mild", "moderate", "severe"]; % Chose a folder containing images of one class.

allFeat = [0;0;0;0;0;0];
allDev = [0;0;0;0;0;0];
featNames = ["Accuracy"; "Sensitivity"; "Specificity"; "Jaccard"; "Dice"; "Correnpondence"];

for currDS = 1 : length(dsType) % iterating over different result folders
    fprintf(1, '%s\n', dsType(currDS));
    %for currOP = 1: length(optimizers)
    for currTG = 1 : length(target) % iterating over the lesions classes

        gtDS = imageDatastore(strcat('groundtruth/dysplasia/', target(currTG)), 'IncludeSubfolders', true);

        jaccIndex = num2cell([0 0]);
        diceCoef = num2cell([0 0]);
        accuracy = num2cell([0 0]);
        sensitivity = num2cell([0 0]);
        specificity = num2cell([0 0]);
        porcentagemAcerto = num2cell([0 0]);
        taxaCorrespondencia = num2cell([0 0]);

        saveDir = strcat('segmentation_eval/', dsType(currDS), '/', target(currTG));
        mkdir(saveDir)

        for currImage = 1 : length(gtDS.Files)

            [gtMask, info] = readimage(gtDS, currImage);
            [filepath,name,ext] = fileparts(info.Filename);

            maskImName = char(strcat('segmentation_results_whole/', dsType(currDS), '/masks/', target(currTG), '/', name, '.png'));

            if isfile(maskImName)

                maskIm = imread(maskImName);

                gtMask2(:,:,1) = gtMask(:,:,1);
                gtMask2(:,:,2) = gtMask(:,:,2);
                gtMask2(:,:,3) = gtMask(:,:,3);

                gtBin = imbinarize(gtMask2(:,:,1));
                resBin = imbinarize(im2uint8(maskIm(:,:,1)));

                %Calculo de falsos e verdadeiros
                adder = gtBin + resBin;
                TP = length(find(adder == 2));
                TN = length(find(adder == 0));
                subtr = gtBin - resBin;
                FP = length(find(subtr == -1));
                FN = length(find(subtr == 1));

                acc = (TP+TN)/(TP+TN+FP+FN); %Accuracy
                sens = TP/(TP+FN); %Sensitivity
                specs = TN/(TN+FP); %specificity
                corrRate = (TP - (0.5 * FP)) / length(find(gtBin == 1)); % Correspondency Rate
                diceCoeff = dice(resBin, gtBin); % Dice Coefficient
                jaccIndexx = jaccard(resBin, gtBin); % Jaccard Index


                %XXXXX GETTING EVERYTHING IN ARRAYS XXXXX
                temp = [cellstr(name) corrRate];
                taxaCorrespondencia = [taxaCorrespondencia; temp]; %Taxa de correspondencia
                temp = [cellstr(name) jaccIndexx];
                jaccIndex = [jaccIndex; temp]; %Coeficiente de Jaccard
                temp = [cellstr(name) diceCoeff];
                diceCoef = [diceCoef; temp]; % Coeficiente DICE
                temp = [cellstr(name) acc];
                accuracy = [accuracy; temp];
                temp = [cellstr(name) sens];
                sensitivity = [sensitivity; temp];
                temp = [cellstr(name) specs];
                specificity = [specificity; temp];
            end
        end
        accuracy(1,:) = [];
        accuracyResult = cell2mat(accuracy(:,2));
        accuracyResult = mean(accuracyResult);
        allFeat(1) = allFeat(1) + accuracyResult;
        stdDev = std(cell2mat(accuracy(:,2)));
        allDev(1) = allDev(1) + stdDev;
        accuracy = [accuracy; cellstr('TOTAL') accuracyResult];
        accuracy = [accuracy; cellstr('STD DEV.') stdDev];
        pathName = strcat(saveDir, '/', 'accuracy', '.csv');
        writetable(cell2table(accuracy), pathName);

        sensitivity(1,:) = [];
        sensitivityResult = cell2mat(sensitivity(:,2));
        sensitivityResult = mean(sensitivityResult);
        allFeat(2) = allFeat(2) + sensitivityResult;
        stdDev = std(cell2mat(sensitivity(:,2)));
        allDev(2) = allDev(2) + stdDev;
        sensitivity = [sensitivity; cellstr('TOTAL') sensitivityResult];
        sensitivity = [sensitivity; cellstr('STD DEV.') stdDev];
        pathName = strcat(saveDir, '/', 'sensitivity', '.csv');
        writetable(cell2table(sensitivity), pathName);

        specificity(1,:) = [];
        specificityResult = cell2mat(specificity(:,2));
        specificityResult = mean(specificityResult);
        allFeat(3) = allFeat(3) + specificityResult;
        stdDev = std(cell2mat(specificity(:,2)));
        allDev(3) = allDev(3) + stdDev;
        specificity = [specificity; cellstr('TOTAL') specificityResult];
        specificity = [specificity; cellstr('STD DEV.') stdDev];
        pathName = strcat(saveDir, '/', 'specificity', '.csv');
        writetable(cell2table(specificity), pathName);

        jaccIndex(1,:) = [];
        jaccardResult = cell2mat(jaccIndex(:,2));
        jaccardResult = mean(jaccardResult);
        allFeat(4) = allFeat(4) + jaccardResult;
        stdDev = std(cell2mat(jaccIndex(:,2)));
        allDev(4) = allDev(4) + stdDev;
        jaccIndex = [jaccIndex; cellstr('TOTAL') jaccardResult];
        jaccIndex = [jaccIndex; cellstr('STD DEV.') stdDev];
        pathName = strcat(saveDir, '/', 'jaccard_index', '.csv');
        writetable(cell2table(jaccIndex), pathName);

        diceCoef(1,:) = [];
        diceResult = cell2mat(diceCoef(:,2));
        diceResult = mean(diceResult);
        allFeat(5) = allFeat(5) + diceResult;
        stdDev = std(cell2mat(diceCoef(:,2)));
        allDev(5) = allDev(5) + stdDev;
        diceCoef = [diceCoef; cellstr('TOTAL') diceResult];
        diceCoef = [diceCoef; cellstr('STD DEV.') stdDev];
        pathName = strcat(saveDir, '/', 'dice_coefficient', '.csv');
        writetable(cell2table(diceCoef), pathName);

        taxaCorrespondencia(1,:) = [];
        corrResult = cell2mat(taxaCorrespondencia(:,2));
        corrResult = mean(corrResult);
        allFeat(6) = allFeat(6) + corrResult;
        stdDev = std(cell2mat(taxaCorrespondencia(:,2)));
        allDev(6) = allDev(6) + stdDev;
        taxaCorrespondencia = [taxaCorrespondencia; cellstr('TOTAL') corrResult];
        taxaCorrespondencia = [taxaCorrespondencia; cellstr('STD DEV.') stdDev];
        pathName = strcat(saveDir, '/', 'taxa_correspondencia', '.csv');
        writetable(cell2table(taxaCorrespondencia), pathName);
    end
    allFeat = allFeat/4;
    allDev = allDev/4;
    pathName = strcat('segmentation_eval_ESWA2/', dsType(currDS), '/', 'allFeat.csv');
    out = cell2table([cellstr(featNames) num2cell(allFeat)]);
    writetable(out, pathName);
    fprintf(1, 'Latex table line: & $%.2f\\pm%.2f$ & $%.2f\\pm%.2f$ & $%.2f\\pm%.2f$ \\\\ \n', allFeat(1)*100, allDev(1)*100, allFeat(6), allDev(6), allFeat(5), allDev(5));
    %fprintf(1, '& %.2f & %.2f & %.2f \\\\ \n', allFeat(1)*100, allFeat(2)*100, allFeat(3)*100);
    allFeat = [0;0;0;0;0;0];
    allDev = [0;0;0;0;0;0];
end
%end
fprintf('XXXXXXXX END XXXXXXXX');
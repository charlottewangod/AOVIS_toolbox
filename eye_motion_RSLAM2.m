clear all;
close all;
clc;

% Load in video
startDir = 'C:\Users\TutenLab-NUC1\Desktop\S-ConeFixation\20114R';
addpath 'C:\Users\TutenLab-NUC1\Box\Enhanced S-cone Syndrome Project\Normal_Subjects_UCB\Code';
vidName = 'C:\Users\TutenLab-NUC1\Desktop\S-ConeFixation\20114R\20114R_yellow.avi';
[~,trialName,~] = fileparts(vidName);
% trialNum = str2num(trialName(end-6:end-4));
vidObj = VideoReader(vidName);

% Load in data file
load('C:\Users\TutenLab-NUC1\Desktop\S-ConeFixation\20114R\20114R_11_4_2022_16_59_4_FixedData.mat');
projector_ppd = 136;
AOSLO_center = 256; % mid point of raster coordinate system
gridSpacing_Deg = 0.5; % Hard coded

AOSLO_ppd = 363.5;
ColorCode = [1 1 0]; %change to other colors for cone type
PlotFixationPRL = 1; %change to 0 for sacPRL
numberOfFramesToInclude = 7;

load('C:\Users\TutenLab-NUC1\Desktop\S-ConeFixation\20114R\20114R_LM_outputs.mat');
t = eye_motions(:,:,1);
x_full=eye_motions(:,:,2);
y_full=eye_motions(:,:,3);
t_new=[t(1,:) t(2,:)+767984 t(3,:)+767984*2 t(4,:)+767984*3 t(5,:)+767984*4 t(6,:)+767984*5];
x_new=[x_full(1,:) x_full(2,:) x_full(3,:) x_full(4,:) x_full(5,:) x_full(6,:)];
y_new=[y_full(1,:) y_full(2,:) y_full(3,:) y_full(4,:) y_full(5,:) y_full(6,:)];
% mod(t, 512);

for v=1:6
x(:,1500*v-1499:1500*v) = reshape(eye_motions(v,:,2), 32, []);
y(:,1500*v-1499:1500*v) = reshape(eye_motions(v,:,3), 32, []);
end

for a=1:32
    for b =1:9000
        if      x(a,b)==inf
                x(a,b)=NaN;
        end
    end
end


x_per_frame = mean(x,'omitnan') +512/2;
% x_per_frame = mean(x, 1)+512/2;

for c=1:32
    for d =1:9000
        if      y(c,d)==inf
                y(c,d)=NaN;
        end
    end
end


% y = reshape(eye_motions(:,:,3), 32, []);
% y_per_frame = mean(y, 1)+512/2;
y_per_frame = mean(y,'omitnan') +512/2;

[map_rows map_columns]= size(retina_map);
saccade_amp=sqrt(diff(x_per_frame).^2 + diff(y_per_frame).^2);
% Find frames where E was presented, and also frames where the subject
% responded
presentFrames = [];
responseFrames = [];

bump = 1;
xPositions_Projector = [];
yPositions_Projector = [];
Precision = [];

for Vid=1:30
    xPositions_Projector = [xPositions_Projector trial_info(Vid).xLoc - projectorSettings.xCenter];
    yPositions_Projector = [yPositions_Projector trial_info(Vid).yLoc - projectorSettings.yCenter];
    Precision = [Precision trial_info(Vid).orientation-trial_info(Vid).response];
    for frameNum = 150*Vid-149:150*Vid
%        trialNum=zeros(vidObj.FrameRate*vidObj.Duration,1);
%       trialNum(frameNum)=Vid;
        testIm = double(read(vidObj,frameNum));
        if std(testIm(end,493:end)) == 0
            if mean(testIm(end,493:end)) == 127
                presentFrames = [presentFrames; frameNum]; %#ok<AGROW>
            elseif mean(testIm(end,493:end)) == 255
                responseFrames = [responseFrames; frameNum]; %#ok<AGROW>
            end
        end
        
    end

                % delete the last present frame
        if length(presentFrames)>length(responseFrames)
        presentFrames(end) = [];
        end

            %delete the last (few) projector orientations
            if length(xPositions_Projector')>length(responseFrames)
        Diff=length(xPositions_Projector')-length(responseFrames);
        xPositions_Projector(end:end+1-Diff) = [];
        yPositions_Projector(end:end+1-Diff) = [];
        Precision(end:end+1-Diff) = [];
            end
end

load('C:\Users\TutenLab-NUC1\Desktop\S-ConeFixation\20217L\20217L_10_24_2022_13_59_7_Data.mat');

for Vid=1:30
    xPositions_Projector = [xPositions_Projector trial_info(Vid).xLoc - projectorSettings.xCenter];
    yPositions_Projector = [yPositions_Projector trial_info(Vid).yLoc - projectorSettings.yCenter];
    Precision = [Precision trial_info(Vid).orientation-trial_info(Vid).response];
    for frameNum = 150*(Vid+30)-149:150*(Vid+30)
%        trialNum=zeros(vidObj.FrameRate*vidObj.Duration,1);
 %       trialNum(150*Vid-149:150*Vid)=Vid+30;
        testIm = double(read(vidObj,frameNum));
        if std(testIm(end,493:end)) == 0
            if mean(testIm(end,493:end)) == 127
                presentFrames = [presentFrames; frameNum]; %#ok<AGROW>
            elseif mean(testIm(end,493:end)) == 255
                responseFrames = [responseFrames; frameNum]; %#ok<AGROW>
            end
        end
        
    end

                % delete the last present frame
        if length(presentFrames)>length(responseFrames)
        presentFrames(end) = [];
        end

            %delete the last (few) projector orientations
            if length(xPositions_Projector')>length(responseFrames)
        Diff=length(xPositions_Projector')-length(responseFrames);
        xPositions_Projector(end:end+1-Diff) = [];
        yPositions_Projector(end:end+1-Diff) = [];
        Precision(end:end+1-Diff) = [];
            end
end

% Compute spacing between grid points in projector and AOSLO pixels
gridSpacing_Pixels_Projector = round(projector_ppd.*gridSpacing_Deg);
gridSpacing_Pixels_AOSLO = AOSLO_ppd.*gridSpacing_Deg;

% use "meshgrid" to create coordinates for plotting
[stimGrid_X_AOSLO,stimGrid_Y_AOSLO] = meshgrid(AOSLO_center-gridSpacing_Pixels_AOSLO:gridSpacing_Pixels_AOSLO:AOSLO_center+gridSpacing_Pixels_AOSLO);
[stimGrid_X_projector, stimGrid_Y_projector] = meshgrid(-projector_ppd*gridSpacing_Deg:projector_ppd*gridSpacing_Deg:projector_ppd*gridSpacing_Deg);
stimGrid_Y_projector = flipud(stimGrid_Y_projector); % Have to flip the Y to get the two coordinate systems to agree

% For each frame, add an index to link to the letter presentation (i.e.
% which of the x and y position in the vector defined in the preceding
% lines). If index = 0, this is the pause where nothing is shown between
% the response and the next presentation
stimIndexVector = zeros(vidObj.FrameRate*vidObj.Duration,1);
PrecisionIndexVector = nan(vidObj.FrameRate*vidObj.Duration,1);
% Cycle through frames and add in the correct index
for n = 1:length(responseFrames)
    stimIndexVector(presentFrames(n):responseFrames(n)) = n;
        if Precision(n) == 0
        PrecisionIndexVector(presentFrames(n):responseFrames(n))=2;
        elseif Precision(n)>0 
        PrecisionIndexVector(presentFrames(n):responseFrames(n))=8;  
        elseif Precision(n)<0 
        PrecisionIndexVector(presentFrames(n):responseFrames(n))=8; 
        end
end

% Now indicate with a yellow spot where the E landed in the raster, and
% watch the eye move accordingly.
showVisualization = 0;
if showVisualization == 1
    figure;
end
numberOfFrames = vidObj.FrameRate*vidObj.Duration;
xPlot = nan(numberOfFrames,1);
yPlot = xPlot;
plotIndexVector = xPlot;
xPlotDisp =nan(numberOfFrames,1);
yPlotDisp =xPlotDisp;

for frameNum = 1:numberOfFrames
    if showVisualization == 1
        testIm = double(read(vidObj,frameNum));
        imshow(testIm./max(testIm(:)));
        hold on;
        title(sprintf('Frame %d', frameNum));
        plot(stimGrid_X_AOSLO, stimGrid_Y_AOSLO, 'wo', 'MarkerSize', 12, 'LineWidth', 2); % show 3x3 stimulus grid;
    end
    % If an E was shown during the frame, figure out where in the raster it
    % was
    if stimIndexVector(frameNum) ~= 0
        % Determine x and y locations, in projector coordinates
        xLoc = xPositions_Projector(stimIndexVector(frameNum));
        yLoc = yPositions_Projector(stimIndexVector(frameNum));
        % Figure out which index corresponds to that coordinate in the
        % AOSLO plotting grid
        plotIndex = intersect(find(stimGrid_X_projector==xLoc), find(stimGrid_Y_projector==yLoc));
        xPlot(frameNum) = stimGrid_X_AOSLO(plotIndex);
        yPlot(frameNum) = stimGrid_Y_AOSLO(plotIndex);
        plotIndexVector(frameNum) = plotIndex;
        if showVisualization == 1
            plot(xPlot(frameNum), yPlot(frameNum), 'yo', 'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'y'); % show stimulus grid;
        end
    end
    
    if showVisualization == 1
        drawnow;
        pause(0.1);
        hold off;
    end
end

figure;
plot(sqrt(diff(x_per_frame).^2 + diff(y_per_frame).^2)*30/AOSLO_ppd);
title('Saccadic Eye Movement deg/s');
ylabel('Speed (degree/sec)');
xlabel('Frame');

for bar = 1:length(responseFrames)
%     if responseFrames(bar)>4499
%     responseFrames(bar)=4499;
%     end
    rectangle('Position', [presentFrames(bar) 0 responseFrames(bar)-presentFrames(bar) 20], 'FaceColor', [1 1 0 .5], 'EdgeColor', 'none');
    %find the peak of each stimulus interval but only look the first half
    %of the interval
    [amp_max(bar), FramesAfterPresent(bar)] =max(saccade_amp(presentFrames(bar):presentFrames(bar)+(responseFrames(bar)-presentFrames(bar))/2));
    saccadeFrames(bar) = FramesAfterPresent(bar)+presentFrames(bar);
end

borderWidth = 256;
figure;

plotFrames = zeros(numberOfFrames,1);

% Set up colormap for color-coded plotting
colorByProjectorPosition = 1; % if zero, will color code according to frame number in the overall video; if one, will color code according to position in the projector array
colorByPrecision = 0;
if colorByProjectorPosition == 1
    cmap = colormap(parula(max(plotIndexVector)));
elseif colorByPrecision == 1
    cmap = colormap(jet(max((PrecisionIndexVector))));
else
    cmap = colormap(jet(numberOfFrames));
end


for n = 1:length(responseFrames)
    if PlotFixationPRL == 1
    plotFrames(responseFrames(n)-numberOfFramesToInclude:responseFrames(n)-2) = 1;
    PRL = 'PRL';
    else
    plotFrames(saccadeFrames(n)+2:saccadeFrames(n)+numberOfFramesToInclude) = 1;
    PRL = 'sacPRL';
%     plotFrames(presentFrames(n):responseFrames(n)) = 1; %plot all frames
    end
end

imshow(retina_map(:,:,:)./255);
for frameNum = 1:vidObj.Duration*vidObj.FrameRate
    hold on;
    line([50 50+AOSLO_ppd/2],[50 50],'color','g');%half degree scale bar
    if plotFrames(frameNum) == 1
        if colorByProjectorPosition == 1
            if ~isnan(plotIndexVector(frameNum))
            plotColor = cmap(plotIndexVector(frameNum),:);
            end
        elseif colorByPrecision == 1
            if ~isnan(PrecisionIndexVector(frameNum))
            plotColor = cmap(PrecisionIndexVector(frameNum),:);
            end
        else
            plotColor = cmap((frameNum),:);
        end
%         plotColor=[0 1 0];
        scatter(xPlot(frameNum)+x_per_frame(frameNum)-256, yPlot(frameNum)+y_per_frame(frameNum)-256, 25, plotColor , 'filled', 'MarkerFaceAlpha', .5 );
        xPlotDisp(frameNum)=xPlot(frameNum)+x_per_frame(frameNum)-256;
        yPlotDisp(frameNum)=yPlot(frameNum)+y_per_frame(frameNum)-256;
    end
%     title(sprintf('Frame %d of %d', frameNum, numberOfFrames));
%     drawnow;
    hold off;
end


% lookBack = min(responseFrames-presentFrames);
% figure; hold on;
% for n = 1:lookBack
% plot(n, nanstd(xPlotDisp(responseFrames-n)), 'ks')
% plot(n, nanstd(yPlotDisp(responseFrames-n)), 'ro');

% scatter(n*ones(length(xPlotDisp(responseFrames-n)),1), (xPlotDisp(responseFrames-n)), 15,'r', 'filled', 'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', .25);
% scatter(n*ones(length(yPlotDisp(responseFrames-n)),1), (yPlotDisp(responseFrames-n)), 15,'b', 'filled', 'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', .25);
% end
% ax = gca;
% ax.XDir = "reverse";
% title('Standard Deviation of Frame Pixel Displacement','FontSize', 12);


presentFramesSec = presentFrames./30;
responseFramesSec = responseFrames./30;


figure; 
subplot(2,1,1), plot(t_new./(30.*512), x_new./(AOSLO_ppd/60), 'r.');
title('x-data');
ylabel('Shift (arcmin');
xlabel('Time (s)');
for bar2 = 1:length(responseFramesSec)
    r1=rectangle('Position', [presentFramesSec(bar2) 0 responseFramesSec(bar2)-presentFramesSec(bar2) 100], 'FaceColor', [1 1 0 .5], 'EdgeColor', 'none');
end
subplot(2,1,2), plot(t_new./(30.*512), y_new./(AOSLO_ppd/60), 'b.');
title('y-data');
ylabel('Shift (arcmin');
xlabel('Time (s)');
hold on;
for bar2 = 1:length(responseFramesSec)
    r2=rectangle('Position', [presentFramesSec(bar2) 0 responseFramesSec(bar2)-presentFramesSec(bar2) 100], 'FaceColor', [1 1 0 .5], 'EdgeColor', 'none');
end

hold off;

figure;
imshow(zeros(map_rows,map_columns));
hold on;
ComputeFixStabilityPerElements_org(xPlotDisp, yPlotDisp, 0.68, 1, ColorCode,0);
hold off;

saveName = [trialName,'_' PRL,'.mat'];
save(saveName);
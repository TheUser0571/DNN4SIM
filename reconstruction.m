clear all
close all
clc
%% importing the necessary paths
addpath('SIM4Expt');
addpath('SIM/Utils');
OTFPath = 'DNN4SIM_data/simulated_sim/OTF.tif';
expPath = 'DNN4SIM_data/simulated_sim_3D-SIM 488_0/AcqDataNoiseless.tif';

%% Loading system OTF file
OTFo = double(imread(OTFPath));
OTFo = OTFpost(OTFo); 

%% Read Expt. Raw SIM Images
aa = loadtiff(expPath);

%% Pre-processing of Raw SIM Images
S1aTnoisy = PreProcessingF(aa(:,:,1));
S2aTnoisy = PreProcessingF(aa(:,:,3));
S1bTnoisy = PreProcessingF(aa(:,:,4));
S1cTnoisy = PreProcessingF(aa(:,:,7));

clear aa

% optional step (may be used if it produces visually better results)
S2aTnoisy = imhistmatch(S2aTnoisy, S1aTnoisy);
S1bTnoisy = imhistmatch(S1bTnoisy, S1aTnoisy);
S1cTnoisy = imhistmatch(S1cTnoisy, S1aTnoisy);

S1aTnoisy = double(S1aTnoisy); 
S2aTnoisy = double(S2aTnoisy); 
S1bTnoisy = double(S1bTnoisy); 
S1cTnoisy = double(S1cTnoisy);

% Create the input for the algorithm
n = 4;
ModFacEst = 1.0.*ones(n, 1);
w = size(OTFo, 1);
wo = w / 2;
Snoisy = zeros(w, w, n);
Snoisy(:, :, 1) = S1aTnoisy;
Snoisy(:, :, 2) = S2aTnoisy;
Snoisy(:, :, 3) = S1bTnoisy;
Snoisy(:, :, 4) = S1cTnoisy;

clear S1aTnoisy S2aTnoisy S1bTnoisy S1cTnoisy

k2a = zeros(n, 2);
PhaseA = zeros(n, 1);
Spattern = zeros(w, w, n);
PSFe = fspecial('gaussian', 14, 1.7);
for i = 1:n    
    S1aTnoisy = Snoisy(:, :, i);
    [k2a(i, :), PhaseA(i)] = PCMseparateF(S1aTnoisy, OTFo, PSFe);    
    Spattern(:, :, i) = PatternCheckF(Snoisy(:,:,i), k2a(i,:), PhaseA(i), ModFacEst(i));
    
end



%% Deconv
u = 202; % width of subimage
uo = u/2;
OTFo = OTFresize(OTFo,u);
k2a = k2a.*(u/w);
[ MaskPetals, doubleSize ] = MaskPetalsF(OTFo,k2a);

PSFe = fspecial('gaussian',16,2.0); % for edgetapering

% position of subimage
xLeft = 200;
yTop = 200;    
%% obtaining the least square solution
[fG1, fG3]  = SIMfreqDeconvAngF(n, ModFacEst, ...
                    OTFo, Snoisy(xLeft+1:xLeft+u, yTop+1:yTop+u, :), ...
                    Spattern(xLeft+1:xLeft+u, yTop+1:yTop+u, :), PSFe, MaskPetals, doubleSize);
%% Determining the object power spectrum
OBJparaA = OBJ4powerPara(fG1,fG3,OTFo, doubleSize);
co = 1.0;
fG1f = W4FilterCenter(fG1,fG3,co,OBJparaA);
G1f = real( ifft2(fftshift(fG1f)) );

%% Plots
v = size(fG1,1);  
h = 20;
figure;
imshow(G1f(h+1:v-h,h+1:v-h),[]) 
title('SR image with artefact')
   
top  = max( max(max(abs(fG1))), max(max(sqrt(abs(fG3)))) );
top  = max( top, max(max(abs(fG1f))) );   
fG1_temp = top*(1-MaskPetals) + abs(fG1);
fG3_temp = top*(1-MaskPetals) + sqrt(abs(fG3));
fG1f_temp = top*(1-MaskPetals) + abs(fG1f);
bottom = min( min(min(fG1_temp)), min(min(sqrt(fG3_temp))) );
bottom = min( bottom, min(min(fG1f_temp)) );
clear fG1_temp fG3_temp fG1f_temp


figure;
surf( log(abs(fG1)), 'EdgeColor','none')
colormap(jet)
axis([0 v 0 v])
box on
caxis manual % This sets the limits of the colorbar to manual for the first plot
caxis([log(bottom) log(top)]);
colorbar;
axis equal
   
figure;
surf( log(sqrt(abs(fG3))) , 'EdgeColor','none')
colormap(jet)
axis([0 v 0 v])
box on
caxis manual % This sets the limits of the colorbar to manual for the first plot
caxis([log(bottom) log(top)]);
colorbar;
axis equal
    
figure;
surf( log(abs(fG1f)) , 'EdgeColor','none')
colormap(jet)
axis([0 v 0 v])
box on
caxis manual % This sets the limits of the colorbar to manual for the first plot
caxis([log(bottom) log(top)]);
colorbar;
axis equal   

% suppressing the spurious frequency-peaks at illumination frequencies
[fG1n, NotchMask] = PeakNotchFilterF(fG1f,k2a);
G1n = real( ifft2(fftshift(fG1n)) );

figure;
imshow(G1n(h+1:v-h,h+1:v-h),[])
title('SR image with artefact suppressed')

figure;
surf( log(abs(fG1n)) , 'EdgeColor','none')
colormap(jet)
axis([0 v 0 v])
box on
caxis manual % This sets the limits of the colorbar to manual for the first plot
caxis([log(bottom) log(top)]);
colorbar;
axis equal 
    
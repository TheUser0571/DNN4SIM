%--------------------------------------------------------------------------
% This script is used to generate the test image reconstructions
% Select the number of the test image and the desired SNR
%--------------------------------------------------------------------------
clear;  close all;% clc
rng(1)
addpath Utils/
%% Parameters
% -- Sample
test_img_nb = 5;
name = ['../DNN4SIM_data/sim_test_img_' num2str(test_img_nb) '.png']; % Name of the file containing the object

% -- PSF
lamb=488;                % Illumination wavelength
res=32;                  % Pixel size (nm)
Na=1.4;                  % Objective numerica aperture
nl=1.51;                 % Refractive index of the objective medium (glass/oil)
ns=1.333;                % Refractive index of the sample medium (water)
type = 0;                % type of PSF (0 : ideal model, 1 : more realistic model)

% -- Patterns 
orr=[2*pi/3 pi/3 0];     % Patterns orientations (vector)   
ph=rand(1,3)*0;            % Phase if set to be random 
a=0.9;                   % Amplitude coefficient of the patterns

% -- Noise
noiseSNR=10;             % SNR of generated data (dB)

% -- Other
displ=2;                 % 0 : only the reconstructed image is displayed
                         % 1 : display also generated data and PSF
                         % 2 : display all intermediate steps of the reconstruction

%% Data Generation
% -- Load image
%x0=double(rgb2gray(imread(name)));x0=x0/max(x0(:)); % rgb
x0=double(imread(name));x0=x0/max(x0(:)); % gray
% When using a .mat file
%{
load '../DNN4SIM_data/dataset_labels_0_3D-SIM 488.mat';
x0 = squeeze(data(1,:,:));
x0 = x0(floor(size(x0,1)/2)-255:floor(size(x0,1)/2)+256,floor(size(x0,2)/2)-255:floor(size(x0,2)/2)+256);
%}

sz=size(x0);

% -- PSF
[PSF,OTF] = GeneratePSF(Na,lamb,sz,res,type,displ>0);

% -- Patterns
k0 =zeros(2,length(orr));
patt =zeros([sz,3]);
for ii=1:length(orr)
    k0(:,ii)=2*pi*ns/lamb*[cos(orr(ii)), sin(orr(ii))]*Na/nl;
    patt(:,:,ii) = GeneratePatterns(k0(:,ii),a*cos(2*ph(ii)),a*sin(2*ph(ii)),sz,res);   
end
patt(:,:,end+1)=rand*ones(sz(1:2));              % add widefield (with unknown intensity)

% -- Data
y = GenerateSIM4data(x0,patt,OTF,noiseSNR,displ>0);          %  Generate 4-SIM data

%% Reconstruction
[x,pattest] = DirectSIM4(y,OTF,res,Na,lamb,1e-3,displ==2);

% Display error on estimated patterns
if displ==2
    for ii=1:3
        figure;
        t1=patt(:,:,ii)-mean(mean(patt(:,:,ii)));
        imagesc(t1-pattest(:,:,ii)*sum(sum(t1.*pattest(:,:,ii)))/norm(pattest(:,:,ii),'fro')^2);
        title(['Error pattern #',num2str(ii)]);colorbar; axis image;
    end
end

% Display Reconstruction
figure;
subplot(1,2,1);imagesc(x); axis image; axis off; title('Reconstructed Image');
subplot(1,2,2);imagesc(log(1+abs(fftshift(fft2(x))))); axis image; axis off; title('FFT Reconstructed Image');
figure;
subplot(1,2,1);imagesc(x0); axis image; axis off; title('Original Image');
subplot(1,2,2);imagesc(log(1+abs(fftshift(fft2(x0))))); axis image; axis off; title('FFT Original Image');
figure;
subplot(1,2,1);imagesc(y(:,:,4)); axis image; axis off; title('WF Image');
subplot(1,2,2);imagesc(log(1+abs(fftshift(fft2(y(:,:,4)))))); axis image; axis off; title('FFT WF Image');

% Save generated images
x = max(0, x);
imwrite(x,['sim_test_img_' num2str(test_img_nb) '_recons_snr' num2str(noiseSNR) '.png'])
wf = y(:,:,4);
wf = max(0,wf);
imwrite(wf,['sim_test_img_' num2str(test_img_nb) '_wf_snr' num2str(noiseSNR) '.png'])

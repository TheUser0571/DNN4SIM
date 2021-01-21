%--------------------------------------------------------------------------
% This script is used to generate the reconstruction from a real SIM image
% Adjust the PSF parameters to the corresponding image parameters
%--------------------------------------------------------------------------
clear;  close all;% clc
addpath Utils/
%% Parameters
% -- Sample
name = '../DNN4SIM_data/real_images/cos7_utubule488.tif';        % Name of the file containing the object
name_wf = '../DNN4SIM_data/real_images/cos7_utubule488_wfMean.tif';
name_orig = '../DNN4SIM_data/real_images/cos7_utubule488_SIM.tif';

% -- PSF
lamb=488;                % Illumination wavelength
res=45;                  % Pixel size (nm)
Na=1.4;                  % Objective numerical aperture
nl=1.51;                 % Refractive index of the objective medium (glass/oil)
ns=1.333;                % Refractive index of the sample medium (water)
type = 1;                % type of PSF (0 : ideal model, 1 : more realistic model)
% -- Other
displ=2;                 % 0 : only the reconstructed image is displayed
                         % 1 : display also generated data and PSF
                         % 2 : display all intermediate steps of the reconstruction

%% Data Generation
% -- Load image
x_orig = double(loadtiff(name_orig)); x_orig=x_orig/max(max(max(x_orig)));
x0=double(loadtiff(name)); x0=x0/max(max(max(x0))); % gray
sz=size(x0);

% -- PSF
[PSF,OTF] = GeneratePSF(Na,lamb,sz,res,type,displ>0);

% -- Generate 4-SIM data
y(:,:,1) = x0(:,:,1);  
y(:,:,2) = x0(:,:,4);
y(:,:,3) = x0(:,:,7); 
wf = double(loadtiff(name_wf)); wf=wf/max(max(max(wf)));
y(:,:,4) = wf;  

%% Reconstruction
[x, pattest] = DirectSIM4(y,OTF,res,Na,lamb,1e-3,displ==2);

% Display Reconstruction
figure;
subplot(1,2,1);imagesc(x); axis image; axis off; title('Reconstructed Image');
subplot(1,2,2);imagesc(log(1+abs(fftshift(fft2(x))))); axis image; axis off; title('FFT Reconstructed Image');
figure;
subplot(1,2,1);imagesc(x_orig); axis image; axis off; title('Original Image');
subplot(1,2,2);imagesc(log(1+abs(fftshift(fft2(x_orig))))); axis image; axis off; title('FFT Original Image');
figure;
subplot(1,2,1);imagesc(y(:,:,4)); axis image; axis off; title('WF Image');
subplot(1,2,2);imagesc(log(1+abs(fftshift(fft2(y(:,:,4)))))); axis image; axis off; title('FFT WF Image');

x = max(0, x);
imwrite(x,'sim_test_img_real_recons.png')

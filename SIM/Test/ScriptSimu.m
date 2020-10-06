clear;clc;close all;

%% General 
sav=1;                                    % to save results
gtpath='Test/object.tif';    % file name ground truth 
expFolder='Test';            % experiment folder

%% PSF
lamb=488;                % Illumination wavelength
res=32;                  % Resolution (nm)
Na=1.4;                  % Objective numerica aperture
nl=1.518;                % Refractive index of the objective medium (glass/oil)
ns=1.333;                % Refractive index of the sample medium (water)

%% Patterns
orr=[0 pi/3 2*pi/3];   % Patterns orientations (vector)                
ph=linspace(0,2*pi,4); % Patterns lateral phases (vector)
ph=ph(1:end-1);  
a=0.9;                 % Amplitude coefficient 
bet=asin(Na/nl);       % Angle between side beams and the optic axis (e.g. bet asin(Na/nl))
wf=0;              	   % Boolean true to add a widefield image in the SIM acquisition

%% Acquisition
downFact=[2 2];  % Downsmpling factor (e.g. [2 2 2]) 
photBud=500;    % Photon Budget

%% Run Simulator
run '../SimuSIM2D.m'

clc; clear; close all;
addpath Utils/
%% Parameters
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
noiseSNR=5;             % SNR of generated data (dB)
%% Data generation and reconstruction

fprintf('Reconstructing ...');
img_count = 0;
for mat_nb = 0:3
    load(['../DNN4SIM_data/DIV2K_' num2str(mat_nb) '.mat']);
    recons_data = zeros(size(data));
    wf_data = zeros(size(data));
    for img_idx = 1:size(data,1)
        fprintf(['\nImage ' num2str(img_count) ' ...']);
        x0 = squeeze(data(img_idx, :, :));
        sz=size(x0);

        % -- PSF
        [PSF,OTF] = GeneratePSF(Na,lamb,sz,res,type,0);

        % -- Patterns
        k0 =zeros(2,length(orr));
        patt =zeros([sz,3]);
        for ii=1:length(orr)
            k0(:,ii)=2*pi*ns/lamb*[cos(orr(ii)), sin(orr(ii))]*Na/nl;
            patt(:,:,ii) = GeneratePatterns(k0(:,ii),a*cos(2*ph(ii)),a*sin(2*ph(ii)),sz,res);   
        end
        patt(:,:,end+1)=rand*ones(sz(1:2));              % add widefield (with unknown intensity)

        % -- Data
        y = GenerateSIM4data(x0,patt,OTF,noiseSNR,0);          %  Generate 4-SIM data
        wf = y(:,:,4);
        wf_data(img_idx, :, :) = max(0, wf);
        % Reconstruction
        [x, pattest] = DirectSIM4(y,OTF,res,Na,lamb,1e-3,0);
        recons_data(img_idx, :, :) = max(0, x);
        
        img_count = img_count + 1;
        
    end
    save(['../DNN4SIM_data/DIV2K_recons_snr' num2str(noiseSNR) '_' num2str(mat_nb)], 'recons_data');
    save(['../DNN4SIM_data/DIV2K_wf_snr' num2str(noiseSNR) '_' num2str(mat_nb)], 'wf_data');
    fprintf('\nFinished reconstructing!\n');
end
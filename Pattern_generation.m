%% General 
sav=1;                                    % to save results
gtpath='SIM/Test/object.tif';             % file name ground truth 
expFolder='DNN4SIM_data/simulated_sim';   % experiment folder

% Image size
sz = [1024,1024];

%% PSF
lamb=488;                % Illumination wavelength
res=32;                  % Resolution (nm)
Na=1.49;                 % Objective numerica aperture
nl=1.518;                % Refractive index of the objective medium (glass/oil)
ns=1.333;                % Refractive index of the sample medium (water)

%% Patterns
orr=[0 pi/3 2*pi/3] + pi/12;   % Patterns orientations (vector)                
ph=linspace(0,pi/4,3); % Patterns lateral phases (vector)
%ph=ph(1:end-1);  
a=0.9;                 % Amplitude coefficient 
bet=asin(Na/nl);       % Angle between side beams and the optic axis (e.g. bet asin(Na/nl))
wf=0;              	   % Boolean true to add a widefield image in the SIM acquisition

%% Acquisition
downFact=[2 2];  % Downsmpling factor (e.g. [2 2 2]) 
photBud=500;    % Photon Budget

% add necessary paths
addpath('SIM/Utils')
run 'SIM/GlobalBioIm-master/setGlobalBioImPath.m'
javaaddpath 'SIM/Utils/PSFGenerator.jar'

%% PSF Generation
fprintf('PSF Generation ...........');
fc=2*Na/lamb*res;
ll=linspace(-0.5,0,sz(1)/2+1);
lr=linspace(0,0.5,sz(1)/2);
[X,Y]=meshgrid([ll,lr(2:end)],[ll,lr(2:end)]);
[th,rho]=cart2pol(X,Y);
OTF=fftshift(1/pi*(2*acos(abs(rho)/fc)-sin(2*acos(abs(rho)/fc))).*(rho<fc));
figure;subplot(1,2,1);imagesc((fftshift(OTF))); axis image; title('OTF'); colormap(fire(200));viscircles(floor(sz/2)+1,fc*sz(1));
psf=real(fftshift(ifft2(OTF)));
subplot(1,2,2);imagesc(psf); axis image; title('PSF'); caxis([0 0.01*max(psf(:))]);
fprintf(' done \n');

%% Patterns Generation
fprintf('Patterns Generation ......');
patt=zeros([sz(1:2),length(orr)*length(ph)]);
[X,Y]=meshgrid(0:sz(2)-1,0:sz(1)-1);X=X*res;Y=Y*res;
it=1;
for ii=1:length(orr)
    k=2*pi*ns/lamb*[cos(orr(ii)), sin(orr(ii))]*sin(bet);
    for jj=1:length(ph)
        patt(:,:,it)=1+ a*cos(2*(k(1)*X+k(2)*Y + ph(jj)));
        it=it+1;
    end
end
if wf, patt(:,:,end+1)=ones(sz(1:2));
end
nbPatt=size(patt,3); % Normalization such that the mean of each pattern is 1/#Patterns
for ii=1:nbPatt
    tmp=patt(:,:,ii);
    patt(:,:,ii)=patt(:,:,ii)/(mean(tmp(:))*nbPatt);
end
figure;subplot(1,2,1);imagesc(patt(:,:,1)); axis image; title('Example pattern');
subplot(1,2,2);imagesc(log(1+abs(fftshift(fftn(patt(:,:,1)))))); axis image; title('Example pattern FFT'); 
viscircles(floor(sz(1:2)/2)+1,fc*sz(1));
fprintf(' done \n');

save([expFolder,'/PSF'],'psf');
save([expFolder,'/OTF'],'OTF');
OTF = imresize(OTF, 0.5);
saveastiff(single(fftshift(OTF)),[expFolder,'/OTF.tif']);
saveastiff(single(psf),[expFolder,'/psf.tif']);
saveastiff(single(patt),[expFolder,'/patterns.tif']);

%% Image Acquisition
ch = '3D-SIM 488'; % Channel

fprintf('Acquisition simulation ...');
img_count = 0;
for mat_nb = 0:3
    load(['DNN4SIM_data/dataset_labels_' num2str(mat_nb) '_' ch '.mat']);
    for img_idx = 1:size(data,1)
        im = squeeze(data(img_idx, :, :));
        % - LinOp Downsampling and integration over camera pixels
        SS=LinOpIdentity(sz);
        S=LinOpDownsample(sz(1:2),downFact);
        % htilde=padarray(ones(downFact),(sz(1:2)-downFact)/2,0,'both');
        % Htilde=LinOpConv(fftn(fftshift(htilde)));
        % - LinOpConv (PSF)
        OTF=Sfft(fftshift(fftshift(psf(:,:,end:-1:1),1),2),3);
        H=LinOpConv(OTF,1,[1 2]);
        % - Acquisition
        acqNoNoise=zeros([S.sizeout,size(patt,3)]);
        fprintf(' Pattern # ');
        for it=1:size(patt,3)
            fprintf([num2str(it),'..']);
            D=LinOpDiag(sz,patt(:,:,it));
        %     acqNoNoise(:,:,it)=S*Htilde*SS*H*D*im;
            acqNoNoise(:,:,it)=S*SS*H*D*im;
        end
        fprintf(' done \n');

        %% Add noise and Save
        for ii=1:length(photBud)
            % - Add Noise
            acq=acqNoNoise;
            if photBud>0
                tmp=sum(acqNoNoise,3);
                factor = photBud(ii)./mean(tmp(:)) ;
                acqNoNoise = acqNoNoise.* factor;
                im = im.*factor;
                acq = random('Poisson',acqNoNoise);
            end
            %     acq=poissrnd(round(acqNoNoise/sum(acqNoNoise(:))*photBud(ii)*prod(sz)));
            %     acqWF=poissrnd(round(acqWFNoNoise/sum(acqWFNoNoise(:))*photBud(ii)*prod(sz)));
            SNR=20*log10(norm(acqNoNoise(:))/norm(acq(:)-acqNoNoise(:)));
            disp(['SNR = ',num2str(SNR),' dB']);

            % - Save
            if sav
                expFolder_ = [expFolder '_' ch '_' num2str(img_count)];
                saveastiff(single(acqNoNoise),[expFolder_,'/AcqDataNoiseless.tif']);
                %saveastiff(single(log(1+abs(fftshift(fftshift(Sfft(acqNoNoise,3),1),2)))),[expFolder_,'/AcqDataNoiseless-FFT.tif']);
                saveastiff(single(acq),[expFolder_,'/AcqData.tif']);
                %saveastiff(single(log(1+abs(fftshift(fftshift(Sfft(acq,3),1),2)))),[expFolder_,'/AcqData-FFT.tif']);
                saveastiff(single(sum(acq,3)),[expFolder_,'/WFData.tif']);
                %saveastiff(single(log(1+abs(fftshift(fft2(sum(acq,3)))))),[expFolder_,'/WFData-FFT.tif']);
                saveastiff(single(sum(acqNoNoise,3)),[expFolder_,'/WFDataNoiseless.tif']);
                %saveastiff(single(log(1+abs(fftshift(fft2(sum(acqNoNoise,3)))))),[expFolder_,'/WFDataNoiseless-FFT.tif']);
            end
        end
        img_count = img_count + 1;
        break % for testing
    end
end
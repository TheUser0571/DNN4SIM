function [x,pattest] = DirectSIM4(y,OTF,res,Na,lamb,w,displ)
%--------------------------------------------------------------------------
% function [x,pattest] = DirectSIM4(y,OTF,res,Na,lamb,w,displ)

% Direct SIM reconstruction from 4 raw images
%
% Inputs : y      : 4-raw data (the last one y(:,:,4) is the widefield image 
%          OTF    : Optical transfert function
%          res    : Pixel size (nm)
%          Na     : Objective numerica aperture
%          lamb   : Illumination wavelength
%          w      : Wiener regul parameter
%          displ  : display parameter (default 0)
%
% Output : x        : reconstructed image
%          pattest  : estimated patterns
%--------------------------------------------------------------------------

%% Pre-computations
OTF0=fftshift(double(OTF>0));   % Ideal OTF
fc=2*Na/lamb*res;               % PSF Cut-off frequency
sz=size(y);sz=sz(1:2);          % Size (x,y)

% -- Initial Deconvolution
% if type
%     w=1e-3;
%     L=zeros(sz);
%     L(floor(sz(1)/2):floor(sz(1)/2)+2,floor(sz(2)/2):floor(sz(2)/2)+2) = [0 -1 0 ; -1 4 -1 ; 0 -1 0];
%     y=zeros(size(y0));
%     for ii=1:4
%         % y(:,:,ii)=real(ifft2(conj(OTF).*fft2(y0(:,:,ii))./(abs(OTF).^2+w))); % *abs(fft2(fftshift(L))).^2
%         y(:,:,ii) = real(ifft2(fft2(deconvlucy(y0(:,:,ii),fftshift(PSF),10)).*fftshift(OTF0)));
%     end
% else
%     y=y0;
% end
% wf=y(:,:,4);


%% Remove WF component to SIM data
wf=y(:,:,4);
fftWF=fftshift(fft2(wf));
G = zeros([sz,3]);
for ii=1:3
    ffty=fftshift(fft2(y(:,:,ii)));
    a =  real(OptWght(ffty,fftWF,sz/2+1,sz(1)/8));
    G(:,:,ii)=real(ifft2(ifftshift(OTF0.*(a*ffty-fftWF))));
end

% if type
%     wf=real(ifft2(fft2(deconvlucy(y(:,:,4),fftshift(PSF),100)).*fftshift(OTF0)));
%     fftWF=fftshift(fft2(wf));
%     %wf=real(ifft2(fft2(x0).*fftshift(OTF0)));
% end

%% Extract patterns parameters 
% Initialize frequencies 
kest=zeros(2,3);
for ii=1:size(G,3)
    kest(:,ii)=GetFreq(G(:,:,ii),fc,res,displ)  ;
end

% Estimate frequencies and phases
ac=zeros(1,3);
as=zeros(1,3);
pattest =zeros([sz,3]);
for ii=1:3
    [kest(:,ii),ac(ii),as(ii)] = GetSIMparams(G(:,:,ii),wf,kest(:,ii),res,OTF,displ);
    pattest(:,:,ii) =  GeneratePatterns(kest(:,ii),ac(ii),as(ii),sz,res);  
    pattest(:,:,ii) = pattest(:,:,ii)-mean(mean(pattest(:,:,ii)));
end
kpx=kest.*sz'*res/pi; % Convert estimated wavevectors to pixels unit

%% Build Masks
% Shifted OTF
OTFm=zeros([sz,3,2]);
for ii=1:3
    OTFm(:,:,ii,1)=imtranslate(OTF0,kpx(:,ii)');
    OTFm(:,:,ii,2)=imtranslate(OTF0,-kpx(:,ii)');
end
% Truncated OTF0
T_OTF0=zeros([sz,3]);
if displ, figure; end
for ii=1:3
    T_OTF0(:,:,ii)=OTF0.*(1-imtranslate(OTF0,2*kpx(:,ii)')).*(1-imtranslate(OTF0,-2*kpx(:,ii)'));
    if displ, subplot(1,3,ii);imagesc(T_OTF0(:,:,ii)); axis image; axis off; colormap gray; title(['Orr #',num2str(ii),' Trunc OTF0']); end
end
% Oval masks
Ovals=zeros([sz,3,2]);
if displ, figure; end
for ii=1:3
    Ovals(:,:,ii,1)=imtranslate(fftshift(OTF).*OTFm(:,:,ii,1),kpx(:,ii)');
    Ovals(:,:,ii,2)=imtranslate(fftshift(OTF).*OTFm(:,:,ii,2),-kpx(:,ii)');
    if displ
        subplot(2,3,ii);imagesc(Ovals(:,:,ii,1)); axis image; axis off; title(['Orr #',num2str(ii),' Oval #1']);
        subplot(2,3,ii+3);imagesc(Ovals(:,:,ii,2)); axis image; axis off; colormap gray; title(['Orr #',num2str(ii),' Oval #2']);
    end
end
% Triangle masks
Tri=zeros([sz,3]);
if displ, figure; end
for ii=1:3
    tmp=OTF0.*prod(1-OTFm(:,:,ii,:),4);
    Tri(:,:,ii)=imtranslate(tmp,kpx(:,ii)')+imtranslate(tmp,-kpx(:,ii)');
end
for ii=1:3
    Tri(:,:,ii)=Tri(:,:,ii).*sum(Tri(:,:,setdiff(1:3,ii)),3).*prod(prod(1-Ovals,4),3);
    if displ, subplot(1,3,ii);imagesc(Tri(:,:,ii)); axis image; axis off; colormap gray;title(['Orr #',num2str(ii),' Tri #1']); end
end

%% Wiener Filter
% TODO  : Prendre en compte proprement l'exp en Fourier due à la phase dans
% la résolution du systeme.
ExtComp = zeros([sz,13]);  % Extracted Component
OtfComp = zeros([sz,13]);  % Corresponding OTF 
comp=1;

% -- Widefield 
ExtComp(:,:,comp)=fftWF.*OTF0;
OtfComp(:,:,comp)=OTF0;
comp=comp+1;

% -- Extract "Oval" components
for ii=1:3
    %tmp=fftshift(fft2(G(:,:,ii)-pattest(:,:,ii).*real(ifft2(ifftshift(fftWF.*T_OTF0(:,:,ii))))));
    tmp=fftshift(fft2(G(:,:,ii)-real(ifft2(OTF.*fft2(pattest(:,:,ii).*real(ifft2(ifftshift(fftWF.*T_OTF0(:,:,ii)))))))));
    
    OtfComp(:,:,comp)=Ovals(:,:,ii,1);
    ExtComp(:,:,comp)=imtranslate(tmp,kpx(:,ii)').*OtfComp(:,:,comp)*2/sqrt(ac(ii)^2+as(ii)^2);    
    comp=comp+1;
    
    OtfComp(:,:,comp)=Ovals(:,:,ii,2);
    ExtComp(:,:,comp)=imtranslate(tmp,-kpx(:,ii)').*OtfComp(:,:,comp)*2/sqrt(ac(ii)^2+as(ii)^2);
    comp=comp+1;
end
% "Rescale intensity"
%  u1=(1-prod(T_OTF0,3)).*OTF0.*ExtComp(:,:,1);
%  u2=(1-prod(T_OTF0,3)).*OTF0.*sum(ExtComp(:,:,2:end),3);
% % a=real(abs(u1(:))'*abs(u2(:))/norm(abs(u1(:)))^2);
%  a=mean(abs(u2(:)))/mean(abs(u1(:)));
% ExtComp(:,:,1)=ExtComp(:,:,1)*a;

% -- Extract "Triangle" components
for ii=1:3
    tmp=fftshift(fft2(G(:,:,ii).*pattest(:,:,ii))).*Tri(:,:,ii)*(2/sqrt(ac(ii)^2+as(ii)^2));
    
    ExtComp(:,:,comp)=ExtComp(:,:,comp)+tmp;
    for jj=setdiff(1:3,ii)
        ExtComp(:,:,comp)=ExtComp(:,:,comp)-imtranslate(tmp,2*kpx(:,jj)');
        ExtComp(:,:,comp)=ExtComp(:,:,comp)-imtranslate(tmp,-2*kpx(:,jj)');
    end
end
ExtComp(:,:,comp)=ExtComp(:,:,comp).*max(Tri,[],3);
OtfComp(:,:,comp)=max(Tri,[],3);

% -- Combination through Wiener filtering
L=zeros(sz);
L(floor(sz(1)/2):floor(sz(1)/2)+2,floor(sz(2)/2):floor(sz(2)/2)+2) = [0 -1 0 ; -1 4 -1 ; 0 -1 0];
x=(real(ifft2(ifftshift(sum(ExtComp.*OtfComp,3)./(sum(OtfComp.^2,3)+w*fftshift(abs(fft2(fftshift(L))).^2))))));
end
function [PSF,OTF] = GeneratePSF(Na,lamb,sz,res,type,displ)
%--------------------------------------------------------------------------
% function [PSF,OTF] = GeneratePSF(Na,lamb,res,type)
%
% Generates the PSF
%
% Inputs : Na    : Numerical Aperture
%          lamb  : Illumination wavelength
%          sz    : size of the returned image
%          res   : Pixel size
%          type  : 0 -> ideal airy PSF 
%                  1 -> more realistic model
%          displ : display (default 0)
%
% Outputs : PSF : image of the PSF
%           OTF : image of the OTF (Fourier of PSF)
%--------------------------------------------------------------------------

if nargin <6
    displ=0;
end

fc=2*Na/lamb*res;        % cut-off frequency
ll=linspace(-0.5,0,sz(1)/2+1);
lr=linspace(0,0.5,sz(1)/2);
[X,Y]=meshgrid([ll,lr(2:end)],[ll,lr(2:end)]);
[~,rho]=cart2pol(X,Y);
if  type == 0
    OTF=fftshift((rho<fc));
else
    OTF=fftshift(1/pi*(2*acos(abs(rho)/fc)-sin(2*acos(abs(rho)/fc))).*(rho<fc));
end
OTF=double(OTF)/max(max(OTF));    
PSF=real(ifft2(OTF));

if displ
    figure;subplot(1,2,1);imagesc(log(1+abs(fftshift(OTF)))); axis image; title('OTF'); viscircles(floor(sz(1:2)/2)+1,fc*sz(1)); colormap(fire(200));
    subplot(1,2,2);imagesc(fftshift(PSF)); axis image; title('PSF'); caxis([0 0.01*max(PSF(:))]);
end
end


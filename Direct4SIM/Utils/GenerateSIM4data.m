function [y] = GenerateSIM4data(x,patt,OTF,noiseSNR,displ)
%--------------------------------------------------------------------------
% function [y] = GenerateSIMdata(x,patt,OTF,noiseLev)

% Generates SIM data
%
% Inputs : x         : Sample 
%          patt      : Patterns
%          OTF       : Optical transfert function
%          noiseSNR  : SNR of the generated data
%          displ     : display (default 0)
%
% Output : y : stack of SIM data
%--------------------------------------------------------------------------

if nargin <5
    displ=0;
end

y=zeros(size(patt));
for ii=1:size(patt,3)
    y_noNoise = real(ifft2(OTF.*fft2(patt(:,:,ii).*x)));
    nn = randn(size(y_noNoise));
    y(:,:,ii) = y_noNoise + norm(y_noNoise(:)) .* 10^(-noiseSNR/20).*nn/norm(nn(:));
end

if displ
    figure;subplot(1,2,1);imagesc(patt(:,:,1)); axis image; title('Example pattern');
    subplot(1,2,2);imagesc(log(1+abs(fftshift(fftn(patt(:,:,1)))))); axis image; title('Example pattern FFT');
    figure;subplot(1,2,1);imagesc(y(:,:,1)); axis image; title('Example acquired data');
    subplot(1,2,2);imagesc(log(1+abs(fftshift(fftn(y(:,:,1)))))); axis image; title('Example acquired data FFT');
end

end

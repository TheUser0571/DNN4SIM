function [y]=SIM_translate(x,k)
%--------------------------------------------------------------------------
% function [y] = SIM_translate(x,k)
%
% Translate an image
%
% Inputs : x         : Sample 
%          k         : Translation vector
%          
%
% Output : y : Translated image
%--------------------------------------------------------------------------
y=fftshift(imtranslate(fftshift(x),round(k)));

end
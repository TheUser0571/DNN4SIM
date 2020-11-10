function [ac,as] = GetPhase(g,wf,k,res,OTF)
%--------------------------------------------------------------------------
% function [ac,as] = GetPhase(g,k,res)
%
% Extract phase from a SIM image knowing the wavevector and the widefield
% image. 
%
% Inputs : g    : SIM image without WF component
%          wf   : widefield image
%          k    : wavevector
%          res  : Pixel size
%
% Output : ac, as : see function GeneratePatterns
%--------------------------------------------------------------------------

% - Pre computations
sz = size(g);              % size

% - Build system
[X,Y]=meshgrid(0:sz(2)-1,0:sz(1)-1); X=X*res; Y=Y*res;
a1 =   wf.* cos(2*(k(1)*X+k(2)*Y));%a1=real(ifft2(OTF.*fft2(a1)));
a2 = - wf.* sin(2*(k(1)*X+k(2)*Y));%a2=real(ifft2(OTF.*fft2(a2)));
A = [a1(:),a2(:)];
AA = A'*A;

% - Solve system and extract phase
s = AA\A'*g(:);
ac=s(1);as=s(2);
end


function [patt] = GeneratePatterns(k,ac,as,sz,res)
%--------------------------------------------------------------------------
% [patt] = GeneratePatterns(orr,ph,sz,res,a)
%
% Generates patterns
%  patt(x,y) = 1 + a*cos(2*(k(1)*x + k(2)*y + ph))
%            = 1 + a*cos(2*ph)*cos(2*(k(1)*x + k(2)*y))
%                - a*sin(2*ph)*sin(2*(k(1)*x + k(2)*y))
%
% Inputs : k    : Wavevector 
%          ac   : quantity a*cos(2*ph)
%          as   : quantity a*sin(2*ph)
%          sz   : size of the returned image
%          res  : Pixel size
%
% Output : patt : stack of images containing the patterns
%--------------------------------------------------------------------------

[X,Y]=meshgrid(0:sz(2)-1,0:sz(1)-1);X=X*res;Y=Y*res;
patt=1+ ac*cos(2*(k(1)*X+k(2)*Y)) - as*sin(2*(k(1)*X+k(2)*Y));
end


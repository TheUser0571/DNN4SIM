function k = GetFreq(y,fc,res,displ)
%--------------------------------------------------------------------------
% k = GetFreq(y,fc,res,displ)
%
% Extract frequency from a SIM image
%
% Inputs : y    : Raw image 
%          fc   : cut-off frequency
%          res  : Pixel size
%          displ : display (default 0)
% 
% Output : k  : estimated k vector 
%--------------------------------------------------------------------------

if nargin <4
    displ=0;
end

% -- Pre-computations
sz = size(y);       % size
y=padarray(y,[sz(1),sz(2)],0);sz = size(y);       
p0=floor(sz/2)+1;   % image center -> p0
% ROI ring between in [cmin*fc, cmax*fc]
% Pattern frequencies peaks should be within this ring
cmin=0.8;           
cmax=1.1;

% -- Mask Fourier Fourier coefficient outside the ring [cmin*fc, cmax*fc]
[I,J]=meshgrid(1:sz(1),1:sz(2));
ringMask=(sqrt((I-sz(1)*0.5).^2+(J-sz(2)*0.5).^2)>cmin*fc*sz(1)).*(sqrt((I-sz(1)*0.5).^2+(J-sz(2)*0.5).^2)<cmax*fc*sz(1));
Fring=ringMask.*abs(fftshift(fft2(y))); 

% -- Detection first peak -> p1
tmp=Fring;tmp(:,1:sz(2)*0.5)=0;
M =max(tmp(:));
[i,j] = find(tmp==M);p1=[i,j];

% -- Detection second peak -> p2
tmp=Fring;tmp(:,sz(2)*0.5+1:end)=0;
M =max(tmp(:));
[i,j] = find(tmp==M);p2=[i,j];

% -- Linear regression between (p0,p1,p2)
P=[p0(2) 1;p1(2) 1;p2(2) 1];yy=[p0(1) p1(1) p2(1)]';
r=(P'*P)\P'*yy;

% TODO: add validation (if fitting error is too large, then problem in
% peaks detection)

% -- Estimated wavevector
k = [1 , r(1)];
k = pi*k/norm(k)*mean([norm(p1-p0),norm(p2-p0)])./(sz*res);

% -- Display
if displ
    figure; hold on; axis ij; imagesc(Fring); axis image; title('Frequencies detection'); %colormap gray;
    plot([p0(2) p1(2) p2(2)],[p0(1) p1(1) p2(1)],'rx','markersize',10,'linewidth',1.5);
    plot(1:sz(2),r(2)+[1:sz(2)]*r(1),'w','linewidth',1.2); axis([1 sz(1) 1 sz(2)])
end
end

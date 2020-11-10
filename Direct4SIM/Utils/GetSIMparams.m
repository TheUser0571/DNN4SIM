function [k,ac,as] = GetSIMparams(y,wf,k0,res,OTF,displ)
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------

if nargin <6
    displ=0;
end

% - Pre computations
rr=5;
sz = size(y);              % size
[X,Y]=meshgrid(0:sz(2)-1,0:sz(1)-1); X=X*res; Y=Y*res;
kPix=k0.*sz'*res/pi;
[I,J]=meshgrid(0:sz(2)-1,0:sz(1)-1); 
OTF0=double(OTF.*fftshift((sqrt((I-kPix(1)-floor(sz(1)/2)-1).^2+(J-kPix(2)-floor(sz(2)/2)-1).^2)<rr)+(sqrt((I+kPix(1)-floor(sz(1)/2)-1).^2+(J+kPix(2)-floor(sz(2)/2)-1).^2)<rr))>0);
y=real(ifft2(fft2(y).*OTF0));
OTF0=OTF0.*OTF;

% -- Initialization
k=k0;
% Phase
A=BuildA(k,wf,X,Y,OTF0);
AA = A'*A;
s = AA\A'*y(:);
% Gradient descent parameters
tau=0.1;
tau_fact=2;
tau_min=1e-10;
nit_k=1;
nit_tot=50;
it=1;
cf(1)=cost(k,wf,s,y,X,Y,OTF0);
if displ
    fig=figure;
end
for jj=1:nit_tot
    % Update Wavevector
    for ii=1:nit_k
        g = grad_k(k,wf,s,y,X,Y,OTF0); % gradient
        ktmp= k -tau*g;
        cf_new=cost(ktmp,wf,s,y,X,Y,OTF0);
        if cf(it)>cf_new
            while cf(it)>cf_new
                k = ktmp;
                tau=tau*tau_fact;
                ktmp = k -tau*g;
                cf_new = cost(ktmp,wf,s,y,X,Y,OTF0);
            end
            tau = tau/tau_fact;
        else
            while cf(it)<=cf_new && (tau>tau_min)
                tau=tau/tau_fact;
                ktmp = k -tau*g;
                cf_new = cost(ktmp,wf,s,y,X,Y,OTF0);
            end
            if (tau>tau_min)
                k=ktmp;
            end
        end
        if (tau<tau_min)
            break;
        end
        it=it+1;      
        cf(it) = cost(k,wf,s,y,X,Y,OTF0);
    end
    if (tau<tau_min) || abs(cf(it)-cf(it-1))/abs(cf(it-1))<1e-5
        break;
    end
    if displ
        figure(fig);
        plot(cf,'linewidth',2); xlim([0 nit_tot*nit_k]);grid;
        title('Freq/Phase Opti: Cv curve')
        set(gca,'FontSIze',14);
        drawnow;
    end
    % Update Phase
    A=BuildA(k,wf,X,Y,OTF0);
    AA = A'*A;
    s = AA\A'*y(:); 
end

ac=s(1);as=s(2);
end

function c = cost(k,wf,s,y,X,Y,OTF0)
     A=BuildA(k,wf,X,Y,OTF0);
     c=0.5*norm(A*s-y(:))^2/numel(y);
end

function A = BuildA(k,wf,X,Y,OTF0)
a1 =   real(ifft2(OTF0.*fft2(wf.* cos(2*(k(1)*X+k(2)*Y)))));
a2 = - real(ifft2(OTF0.*fft2(wf.* sin(2*(k(1)*X+k(2)*Y)))));
A = [a1(:),a2(:)];
end

function g = grad_k(k,wf,s,y,X,Y,OTF0)
    A=BuildA(k,wf,X,Y,OTF0);
    g(1)=-(A*[s(2);-s(1)].*2.*X(:))'*(A*s-y(:))/numel(y);
    g(2)=-(A*[s(2);-s(1)].*2.*Y(:))'*(A*s-y(:))/numel(y);
    g=g';
end
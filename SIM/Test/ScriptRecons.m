clear;%close all;%clc;

%% Parameters
% -- Paths
dataname='AcqData.tif';   % file name data image
psfname='PSF';            % file name psf
pattname='patterns.tif';  % file name patterns
gtname='object.tif';      % file name ground truth (put [] if not used)
outFolder='';
sav=1;     % Boolean if true save the result
disp(dataname);

% -- Data
valback=0;                % Background value
nbPatt=9;                 % Number of patterns
downFact=[2 2];           % Downsampling factors
apodize=0;                % Boolean true to apodize the data
useWF=0;
selPatt=[];

% -- Objective Function
lamb=5e-4;                % Regularization parameter for TV
Reg=1;                    % Choice regul: 1 for TV, 2 for Hessian-Schatten
nbOutSl=0;                % Number of considered out-of-focus slices
symPsf= 0;                % Boolean true if psf is symmetric  (in this case 2*nbOutSl out-of-focus slides are considered on the same side of the psf in z)


% -- SIM reconstruction
maxIt= 200;                   % Max iterations
alg=1;                        % Algorithm (1: ADMM / 2: Primal-Dual)
ItUpOut=round(maxIt/10);      % Iterations between to call to OutputOpti

% ADMM
rhoDTNN= 1e-2;       % rho parameter (ADMM) associated to data term
rhoReg= 1e-2;        % rho parameter (ADMM) associated to the regularization (must be greater or equal than rhoDTNN if iterCG=0)
split=3;             % Splitting strategy for data fidelity:
%    0: no-splitting
%    1: u_p=SHW_px
%    2: u_p=HW_px
%    3: u_p=W_px (reformulation proposed in [1] is used)
iterCG=0;         % Max number of conjugate gradient iterations for splittings (0,1 and 2)
valId=2;          % Scaling (>1) of the identity operator for the reformulation in [1] (only for splitting 3)

% Primal-Dual
%tau=            -> tau parameter (Primal-Dual)
%rho=            -> rho parameter (Primal-Dual) in ]0,2[
%split=          -> Splitting strategy for data fidelity:
%                       0: no-splitting
%                       1: u_p=SHW_px
%                       2: u_p=HW_px
%                       3: u_p=W_px

%% Run script
run '../SimScript2D.m'

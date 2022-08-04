%% Study case: Clamped-free beam
% The dynamic response of a 100 m high clamped-free steel beam is studied.
% Simulated time series are used, where the first three eigenmodes have been
% taken into account.
clearvars;close all;clc;
load('data.mat')
fn = wn/(2*pi);
Nmodes = numel(fn);
fs = 1/median(diff(t));
rng(1)
%% Manual procedure

tic
[phi_FDD,fn_FDD,zeta] = AFDD(Az,t,Nmodes,'PickingMethod','auto');
toc
% we plot the mode shapes
figure
for ii=1:size(phi_FDD)
    subplot(2,2,ii)
    hold on;box on;
    h1 = plot(linspace(0,1,size(phi_FDD,2)),phi_FDD(ii,:),'ro','linewidth',1.5);
    h2 = plot(linspace(0,1,size(phi,2)),-phi(ii,:),'k-','linewidth',1.5);
    xlabel('$y$ (a.u.)','interpreter','latex')
    ylabel(['$\phi_',num2str(ii),'$'],'interpreter','latex')
    if ii==1
        legend('Measured','Target','location','SouthWest')
    end
end
% The theoretical and measured eigenfrequencies agrees well !
disp('left: target eigen frequencies. Right: Measured eigenfrequencies')
disp([fn(:),fn_FDD(1:Nmodes)'])
disp('left: target damping. Right: Measured damping')
disp([5e-3*ones(Nmodes,1),zeta(:),])

%% Automated procedure 1: minimalist example
%  Minimalist example with automated procedure for those who don't want to
%  read too much

[phi_FDD,fn_FDD,zeta] = AFDD(Az,t,Nmodes);

% plot the mode shapes
figure
for ii=1:size(phi_FDD)
    subplot(2,2,ii)
    hold on;box on;
    h1 = plot(linspace(0,1,size(phi_FDD,2)),phi_FDD(ii,:),'gd','linewidth',1.5);
    h2 = plot(linspace(0,1,size(phi,2)),-phi(ii,:),'k-','linewidth',1.5);
    xlabel('$y$ (a.u.)','interpreter','latex')
    ylabel(['$\phi_',num2str(ii),'$'],'interpreter','latex')
    if ii==1
        legend('Measured','Target','location','SouthWest')
    end
end

% Comparison between the measured and target eigen freq. and mode shapes
disp('left: target eigen frequencies. Right: Measured eigenfrequencies')
disp([fn(:),fn_FDD(1:Nmodes)'])
disp('left: target damping. Right: Measured damping')
disp([5e-3*ones(Nmodes,1),zeta(:),])

%% Automated procedure 2: 2-step analysis

% First step: determination of the eigenfrequencies
[~,fn_FDD,zeta] = AFDD(Az(1:5:end,:),t,Nmodes,'dataPlot',1);
% we show that the estimated zeta is ca. 10 x larger for the first 2 modes
% than expected.
% the theoritical and measured eigen frequencies agrees however well!
disp('left: target eigen frequencies. Right: Measured eigenfrequencies')
disp([fn(:),fn_FDD(1:Nmodes)'])

% Second step: determination of the modal damping ratio
% We use a high value for M and prescribed eigenfrequencies
% We use the option 'dataPlot' to plot intermediate figures, to illustrate
% the method, and to check the accuracy of the results.
[phi_FDD,fn_FDD,zeta] = AFDD(Az,t,Nmodes,'fn',fn_FDD,'M',8192);
% Plot the mode shapes
figure
for ii=1:size(phi_FDD)
    subplot(2,2,ii)
    hold on;box on;
    h1 = plot(linspace(0,1,size(phi_FDD,2)),phi_FDD(ii,:),'csq','linewidth',1.5);
    h2 = plot(linspace(0,1,size(phi,2)),-phi(ii,:),'k-','linewidth',1.5);
    xlabel('$y$ (a.u.)','interpreter','latex')
    ylabel(['$\phi_',num2str(ii),'$'],'interpreter','latex')
    if ii==1
        legend('Measured','Target','location','SouthWest')
    end
end
disp('left: target damping. Right: Measured damping')
disp([5e-3*ones(Nmodes,1),zeta(:),])


%% Case of user-defined boundaries for the selected peaks.
% The boundaries for the selected peaks (lines 209 in the main function AFDD)
% may not be adapted if the eigenfrequency values range from low to high
% frequencies. For this reason, it is possible to manually give the upper
% boundaries (UB) and the lower boundaries(LB) as shown below for the first 4
% eigenfrequencies of the beam studied:

% lower boundary for the first four modes (Default: LB = 0.9*fn)
LB = [0.15,0.9,2.8,5.5];
% upper boundary boundary for the first four modes (Default: UB = 0.9*fn)
UB = [0.18,1.15,3.1,6.1];

% Visualization of the boundaries
clf;close all;
figure
plot(1:4,fn,'ko-',1:4,LB,'r-',1:4,UB,'b-')
ylabel('$f_n$ (Hz)','interpreter','latex')
legend('Measured eigen frequency','user-defined lower boundary','user-defined upper boundary','location','best')
xlabel('Mode number')
set(gca,'xtick',[1,2,3,4])

% Calculation of the modal parameters with user-defined UBs and LBs
[phi_FDD,fn_FDD,zeta] = AFDD(Az,t,Nmodes,'M',8192,'UB',UB,'LB',LB);

figure
for ii=1:size(phi_FDD)
    subplot(2,2,ii)
    hold on;box on;
    h1 = plot(linspace(0,1,size(phi_FDD,2)),phi_FDD(ii,:),'msq','linewidth',1.5);
    h2 = plot(linspace(0,1,size(phi,2)),-phi(ii,:),'k-','linewidth',1.5);
    xlabel('$y$ (a.u.)','interpreter','latex')
    ylabel(['$\phi_',num2str(ii),'$'],'interpreter','latex')
    if ii==1
        legend('Measured','Target','location','SouthWest')
    end
end
disp('left: target damping. Right: Measured damping')
disp([5e-3*ones(Nmodes,1),zeta(:),])
    
    
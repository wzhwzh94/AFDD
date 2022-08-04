function [newS,newF,fn] = getS_from_FDD(Az,t,M,Nmodes)
%% Pre-processing
[Nyy,N] = size(Az);
fs = 1/median(diff(t));
if M>numel(t),    error('M must be less than numel(t)');end
%% Computation of the spectral matrix G
%  size(G) is [N x Nyy x Nyy]
if rem(M,2),
    G = zeros(Nyy,Nyy,round(M/2));
else
    G = zeros(Nyy,Nyy,round(M/2)+1);
end
for ii=1:Nyy,
    for jj=1:Nyy,
        [G(ii,jj,:),f] = cpsd(Az(ii,:),Az(jj,:),M,round(M/2),M,fs);
    end
end
%% Application of SVD to G
% U =zeros(size(G));
S =zeros(Nyy,size(G,3));
% V =zeros(size(G));
for ii=1:size(G,3),
    [~,diagMat,~] = svd(G(:,:,ii));
    S(:,ii) = diag(diagMat);
end

S = S(1,:);



% interpolation to improve accuracy of peak picking and damping estimation
Ninterp=5;
newF = linspace(f(1),f(end),Ninterp*numel(f));
newS = interp1(f,S(1,:),newF,'pchip');
newS = newS./max(newS); % normalized power spectral density

indMax = manualPickPeaking(newF,newS,Nmodes);
fn = newF(indMax);

function [Fp] = manualPickPeaking(f,S,Nmodes)
                %%
        display('Peak selection procedure')
        display('a: Draw rectangles around peaks while holding left click')
        display('b: Press "Space" key to continue the peak selection')
        display('c: Press "any other key" if you have selected a peak by mistake and want to ignore it')
        
        clf
        semilogx(f,mag2db(S))
        grid on
        xlim([f(2),f(end)])
        ylim([min(mag2db(S)),max(mag2db(10*S))])
        hold on
        xlabel('Frequency (Hz)')
        ylabel('1st Singular values of the PSD matrix (db)')
        Fp=[];% Frequencies related to selected peaks
        while numel(Fp)<Nmodes
            myRec=getrect;                                                                          % Draw a rectangle around the peak
            [~,P1]=min(abs(f-myRec(1)));
            [~,P2]=min(abs(f-(myRec(1)+myRec(3))));
            [~,P3]=max(S(P1:P2));
            indPeak=P3+P1-1;                                                                         % Frequency at the selected peak
            scatter(f(indPeak),mag2db(S(indPeak)),'MarkerEdgeColor','b','MarkerFaceColor','b')         % Mark this peak
            pause;
            key=get(gcf,'CurrentKey');
            if strcmp(key,'space'),
                % Press space to continue peak selection
                Fp=[Fp,indPeak];
                scatter(f(indPeak),mag2db(S(indPeak)),'MarkerEdgeColor','g','MarkerFaceColor','g')      % Mark this peak as green
            else
                % Press any other key to ignore this peak
                scatter(f(indPeak),mag2db(S(indPeak)),'MarkerEdgeColor','r','MarkerFaceColor','r')      % Mark this peak as red
            end
        end
        % Number selected peaks, respectively
        Fp=sort(Fp);
        pause(0.01);
    end
end

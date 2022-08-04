function [peaks,criterion] = pickpeaks(V,select,display)
% -------------------------------------------------------------
% Scale-space peak picking
% ------------------------
% This function looks for peaks in the data using scale-space theory. 
% 
% input : 
%   * V : data, a vector
%   * select : either: 
%       - select >1 : the number of peaks to detect
%       - 0<select<1 : the threshold to apply for finding peaks 
%         the closer to 1, the less peaks, the closer to 0, the more peaks
%   * display : whether or not to display a figure for the results. 0 by
%               default
%   * ... and that's all ! that's the cool thing about the algorithm =)
% 
% outputs :
%   * peaks : indices of the peaks
%   * criterion : the value of the computed criterion. Same
%                 length as V and giving for each point a high value if
%                 this point is likely to be a peak
% 
% The algorithm goes as follows:
% 1°) set a smoothing horizon, initially 1;
% 2°) smooth the data using this horizon
% 3°) find local extrema of this smoothed data
% 4°) for each of these local extrema, link it to a local extremum found in
%     the last iteration. (initially just keep them all) and increment the 
%     corresponding criterion using current scale. The
%     rationale is that a trajectory surviving such smoothing is an important
%     peak
% 5°) Iterate to step 2°) using a larger horizon.

%data is a vector
V = V(:)-min((V(:)));

%input parsin
if nargin < 3
    display=0;
end
if nargin < 2
    select= 0;
end

n = length(V);

%definition of local variables
buffer = zeros(n,1);
criterion = zeros(n,1);
if select < 1
    minDist = n/20;
else
    minDist = n/select;
end
%horizons = round(linspace(1,ceil(n/20),50));
horizons = unique(round(logspace(0,2,50)/100*ceil(n/20)));

%horizons=1:2:50;
Vorig = V;

% all this tempMat stuff is to avoid calling findpeaks which is horribly
% slow for our purpose
tempMat = zeros(n,3);
tempMat(1,1)=inf;
tempMat(end,3)=inf;

% loop over scales
for is=1:length(horizons)
    
    %sooth data, using fft-based convolution with a half sinusoid
    horizon = horizons(is);
    if horizon > 1
        w=max(eps,sin(2*pi*(0:(horizon-1))/2/(horizon-1)));
        w=w/sum(w);    
        %V=conv(V(:),w(:),'same');
        V = real(ifft(fft(V(:),n+horizon).*fft(w(:),n+horizon)));
        V = V(1+floor(horizon/2):end-ceil(horizon/2));
    end

    %find local maxima
    tempMat(2:end,1) = V(1:end-1);
    tempMat(:,2) = V(:);
    tempMat(1:end-1,3) = V(2:end);
    [useless,posMax] =max(tempMat,[],2);
    I = find(posMax==2);
    I = I(:)';
    
    %initialize buffer
    newBuffer = zeros(size(buffer));
    
    if is == 1
        % if first iteration, keep all local maxima
        newBuffer(I) = Vorig(I);
    else    
        old = find(buffer);
        old = old(:)';
        if isempty(old)
            continue;
        end
        [c,p] = sort(old);
        [useless,ic] = histc(I,[-inf,(c(1:end-1)+c(2:end))/2,inf]);
        iOld = p(ic);
        d = abs(I-old(iOld));
 
        %done, now select only those that are sufficiently close
        neighbours = iOld(d<minDist);

        if ~isempty(neighbours)
            newBuffer(old(neighbours)) = V(old(neighbours))*is^2;
        end
    end
    %update stuff
    buffer = newBuffer;
    criterion = criterion + newBuffer;
end

%normalize criterion
criterion = criterion/max(criterion);

%find peaks based on criterion
if select<1
    peaks = find(criterion>select);
else
%     sorted = find(criterion>1E-3);
%     [~,order] = sort(criterion(sorted),'descend');
%     peaks = sorted(order(1:min(length(sorted),select)));
    [useless,order] = sort(criterion,'descend');
    peaks = order(1:select);
end

if display
    %display
    clf
    plot(Vorig,'LineWidth',2);
    hold on
    plot(criterion*max(Vorig),'r');
    hold on
    plot(peaks,Vorig(peaks),'ro','MarkerSize',10,'LineWidth',2)
    grid on
    title('Scale-space peak detection','FontSize',16);
    legend('data','computed criterion','selected peaks');
end
end


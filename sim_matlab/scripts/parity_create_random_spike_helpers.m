function varargout = parity_create_random_spike_helpers(action, varargin)
%PARITY_CREATE_RANDOM_SPIKE_HELPERS Wrapper for local helpers inside create_random_spike.m.

setup_repro_paths();

switch action
    case 'create_random_spike_no_lfp'
        N = varargin{1};
        F = varargin{2};
        sf = varargin{3};
        P = varargin{4};
        force_active = varargin{5};
        U = varargin{6};
        activation_choice = varargin{7};
        [S, LFP] = parity_create_random_spike_no_lfp(N, F, sf, P, force_active, U, activation_choice);
        varargout{1} = S;
        varargout{2} = LFP;
    case 'create_sine'
        lfp = varargin{1};
        sf = varargin{2};
        F = varargin{3};
        phase_offsets = varargin{4};
        [LFP, fP] = parity_create_sine(lfp, sf, F, phase_offsets);
        varargout{1} = LFP;
        varargout{2} = fP;
    otherwise
        error('Unknown action: %s', action);
end
end

function [S,LFP]=parity_create_random_spike_no_lfp(N,F,~,P,force_active,U,activation_choice)
LFPp=ones(1,F);
LFP=ones(1,F);
for i=1:N
    p=P(i);
    bin=size(LFPp,2)/F;
    p=p/bin;
    p=p.*LFPp;
    temp=double((U(i,:)-p)<0);
    if force_active==1
        if sum(temp)==0
            ix=1:numel(p);
            ix=ix(p>0);
            pnt=ix(activation_choice(i));
            temp(pnt)=1;
        end
    end
    S(i,:)=temp; %#ok<AGROW>
end
S=bin_data(S,1,bin);
end

function [LFP,fP]=parity_create_sine(lfp,sf,F,phase_offsets)
lfp_sf=120;
Freq=lfp-4:0.01:lfp+5;
LFP=zeros(1,(F/sf)*lfp_sf);
dt = 1/lfp_sf;
StopTime = (F/lfp_sf);
t = (dt:dt:StopTime*(lfp_sf/sf));
for i=1:numel(Freq)
    LFP_t= sin(2.*pi.*Freq(i).*t+phase_offsets(i));
    LFP=LFP+LFP_t*normpdf(Freq(i),lfp,1);
end
[yupper,ylower] = envelope(LFP);
fP=(LFP-ylower)./(yupper-ylower);
end

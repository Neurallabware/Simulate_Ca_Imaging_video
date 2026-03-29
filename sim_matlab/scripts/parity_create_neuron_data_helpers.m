function varargout = parity_create_neuron_data_helpers(action, varargin)
%PARITY_CREATE_NEURON_DATA_HELPERS Wrapper for local helpers inside create_neuron_data.m.

setup_repro_paths();

switch action
    case 'create_spatial'
        BW = varargin{1};
        X = varargin{2};
        Y = varargin{3};
        ix = varargin{4};
        opt = parity_cnd_opt(varargin{5:end});
        [Aout, bA] = parity_create_spatial(BW, opt, X, Y, ix);
        varargout{1} = Aout;
        varargout{2} = bA;
    case 'conv_spikes'
        S = varargin{1};
        invtRise = varargin{2};
        invtDecay = varargin{3};
        rise_draws = varargin{4};
        decay_draws = varargin{5};
        F = varargin{6};
        varargout{1} = parity_conv_spikes(S, invtRise, invtDecay, rise_draws, decay_draws, F);
    case 'fix_inactive'
        S = varargin{1};
        varargout{1} = parity_fix_inactive(S);
    otherwise
        error('Unknown action: %s', action);
end
end

function opt = parity_cnd_opt(varargin)
inp = inputParser;
addParameter(inp, 'd', [220, 300]);
addParameter(inp, 'min_dist', 8);
addParameter(inp, 'Nneu', 50);
addParameter(inp, 'plotme', 0);
addParameter(inp, 'ses', 2);
addParameter(inp, 'seed', 1);
addParameter(inp, 'CA1_A', false);
addParameter(inp, 'A2', false);
addParameter(inp, 'scale', 1);
parse(inp, varargin{:});
opt = inp.Results;
rng(opt.seed, 'twister');
end

function [Aout,bA]=parity_create_spatial(BW,opt,X,Y,ix)
if opt.CA1_A
    load(resolve_repo_data_file('A_CA1.mat'));
elseif opt.A2
    load(resolve_repo_data_file('A2.mat'));
else
    load(resolve_repo_data_file('A.mat'));
end
N=size(X,2);
if size(A,4)>1
    w=linspace(0,1,opt.ses);
    for i=1:length(w)
        As(:,:,:,i)=squeeze(A(:,:,1,ix))*(1-w(i))+squeeze(A(:,:,2,ix))*(w(i));
    end
else
    As=repelem(A(:,:,ix),1,1,1,opt.ses);
end
if opt.scale>1
    [d1, d2, s, c] = size(As);
    temp = zeros(round(d1*opt.scale), round(d2*opt.scale), s, c, 'like', As);
    for si = 1:s
        for ci = 1:c
            temp(:,:,si,ci) = imresize(As(:,:,si,ci), opt.scale, 'bicubic');
        end
    end
    As=temp;
    clear temp
end
for s=1:opt.ses
    Aout_t=[];
    bA_t=[];
    for i=1:length(X)
        Z=zeros(opt.d);
        Z(Y(i),X(i))=1;
        Aout_t(:,:,i)=conv2(Z, As(:,:,i,s), 'same');
        bA_t(:,:,i)=imgaussfilt(Aout_t(:,:,i),10,'FilterSize',opt.d(2)+1);
    end
    Aout_t=Aout_t./max(Aout_t,[],'all');
    bA_t=bA_t./max(bA_t,[],'all');
    Aout{s}=Aout_t; %#ok<AGROW>
    bA{s}=bA_t; %#ok<AGROW>
end
end

function out = parity_conv_spikes(S,invtRise,invtDecay,rise_draws,decay_draws,F)
t = 0:F-1;
for i=1:size(S,1)
    rise=1/rise_draws(i);
    decay=1/decay_draws(i);
    g = exp(-t/decay)-exp(-t/rise);
    temp=S(i,:);
    c = conv(temp, g);
    out(i,:)=c(1,1:F); %#ok<AGROW>
end
end

function S = parity_fix_inactive(S)
d=size(S,2);
in=find(sum(S,2)==0);
for i=1:size(in,1)
    S(in(i),randi(d,1))=1;
end
end

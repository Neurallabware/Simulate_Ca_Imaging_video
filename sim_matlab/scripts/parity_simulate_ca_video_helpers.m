function varargout = parity_simulate_ca_video_helpers(action, varargin)
%PARITY_SIMULATE_CA_VIDEO_HELPERS Wrapper for local helpers inside Simulate_Ca_video.m.

setup_repro_paths();

switch action
    case 'int_var'
        varargout{1} = parity_int_var(varargin{:});
    case 'get_out_path'
        opt = parity_int_var(varargin{:});
        outpath = parity_get_out_path(opt);
        if isempty(outpath)
            outpath = '';
        end
        varargout{1} = outpath;
    case 'create_baseline'
        opt = parity_int_var(varargin{:});
        varargout{1} = parity_create_baseline(opt);
    case 'add_remapping_drifting'
        c = varargin{1};
        opt = parity_int_var(varargin{2:end});
        varargout{1} = parity_add_remapping_drifting(c, opt);
    case 'make_xy_motion_ar'
        [dx, dy] = parity_make_xy_motion_ar(varargin{:});
        varargout{1} = dx;
        varargout{2} = dy;
    case 'make_xy_motion_ar_from_innov'
        [dx, dy] = parity_make_xy_motion_ar_from_innov(varargin{:});
        varargout{1} = dx;
        varargout{2} = dy;
    case 'apply_xy_translation'
        varargout{1} = parity_apply_xy_translation(varargin{:});
    otherwise
        error('Unknown action: %s', action);
end
end

function opt = parity_int_var(varargin)
inp = inputParser;
valid_v = @(x) isnumeric(x);
valid_bool = @(x) islogical(x) || (isnumeric(x) && isscalar(x) && ismember(x,[0,1]));
valid_batch = @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0);
addParameter(inp, 'outpath', [])
addParameter(inp, 'min_dist', 8, valid_v)
addParameter(inp, 'Nneu', 50, valid_v)
addParameter(inp, 'PNR', 2, valid_v)
addParameter(inp, 'd', [220, 300], valid_v)
addParameter(inp, 'F', 1000, valid_v)
addParameter(inp, 'batch_sz', [], valid_batch)
addParameter(inp, 'overlap', 1, valid_v)
addParameter(inp, 'overlapMulti', 0, valid_v)
addParameter(inp, 'NR_misalignment', 0, valid_v)
addParameter(inp, 'motion_sz', 60, valid_v)
addParameter(inp, 'translation_misalignment', 1, valid_v)
addParameter(inp, 'session_motion_std', 3, valid_v)
addParameter(inp, 'session_motion_phi', 0.99, valid_v)
addParameter(inp, 'session_motion_max', [], @(x) isnumeric(x) || isempty(x))
addParameter(inp, 'session_motion_seed', [], valid_v)
addParameter(inp, 'ses', 2, valid_v)
addParameter(inp, 'seed', 'shuffle')
addParameter(inp, 'B', '1')
addParameter(inp, 'spike_prob', [-4.9, 2.25])
addParameter(inp, 'save_videos', true, valid_bool)
addParameter(inp, 'create_mask', true)
addParameter(inp, 'comb_fact', 0, valid_v)
addParameter(inp, 'drift', 0, valid_v)
addParameter(inp, 'LFP', 0, valid_v)
addParameter(inp, 'sf', 10, valid_v)
addParameter(inp, 'plotme', 0, valid_v)
addParameter(inp, 'invtRise', [2.08, 0.29], valid_v)
addParameter(inp, 'invtDecay', [0.55, 0.44], valid_v)
addParameter(inp, 'disappearBV', 0, valid_v)
addParameter(inp, 'CA1_A', false)
addParameter(inp, 'A2', false)
addParameter(inp, 'force_active', 1)
addParameter(inp, 'save_avi', 1)
addParameter(inp, 'scale', 1)
addParameter(inp, 'save_mat', 1, valid_bool)
addParameter(inp, 'save_GT', 1, valid_bool)
inp.KeepUnmatched = true;
parse(inp, varargin{:});
opt = inp.Results;
opt.B = str2num(string(opt.B)); %#ok<ST2NM>
if isfield(inp.Unmatched,'save_files') && any(strcmpi(inp.UsingDefaults,'save_videos'))
    opt.save_videos = logical(inp.Unmatched.save_files);
end
if isfield(inp.Unmatched,'Nneu2')
    opt.Nneu2 = inp.Unmatched.Nneu2;
end
opt.save_videos = logical(opt.save_videos);
opt.save_mat = logical(opt.save_mat);
opt.save_GT = logical(opt.save_GT);
if isempty(opt.batch_sz)
    opt.batch_sz = opt.F;
end
opt.batch_sz = max(1, min(opt.F, round(opt.batch_sz)));
s = rng(opt.seed,'twister');
s = rng(opt.seed,'twister');
opt.seed = s.Seed;
while 1
    s = rng(opt.seed,'twister');
    if s.Seed==opt.seed
        break
    end
end
end

function outpath = parity_get_out_path(opt)
if opt.save_videos || opt.save_mat || opt.save_GT
    if isempty(opt.outpath)
        outpath = pwd;
    else
        outpath = opt.outpath;
    end
else
    outpath = [];
end
end

function bl = parity_create_baseline(opt)
comb_fact = 1-opt.comb_fact;
ses = opt.ses;
load(resolve_repo_data_file('BL.mat'),'Mr');
if length(opt.B)==1
    Mr = Mr{opt.B,1};
    bl = zeros(opt.d(1),opt.d(2),2);
    for i=1:size(Mr,3)
        bl(:,:,i) = imresize(Mr(:,:,i),opt.d);
    end
    if ses>2
        p = rand(1,ses);
        temp = bl;
        bl = zeros(opt.d(1),opt.d(2),ses);
        for i=1:ses
            bl(:,:,i) = temp(:,:,1)*p(i)+temp(:,:,2)*(1-p(i))+randn/10;
        end
        bl = mat2gray(bl);
    end
else
    temp = cat(3,imresize(Mr{opt.B(1),1}(:,:,1),opt.d),imresize(Mr{opt.B(2),1}(:,:,1),opt.d));
    p = linspace(1,comb_fact,ses);
    for i=1:ses
        bl(:,:,i) = temp(:,:,1)*p(i)+temp(:,:,2)*(1-p(i));
    end
end
if opt.disappearBV>0
    for i=1:size(bl,3)
        im = bl(:,:,i);
        gs = medfilt2(mean(im,2)*mean(im,1),[25 25])+rescale(randn(size(im)),min(im,[],'all'),max(im,[],'all'))./10;
        im = rescale(im*(1-opt.disappearBV)+gs*opt.disappearBV,min(bl(:,:,i),[],'all'),max(bl(:,:,i),[],'all'));
        bl(:,:,i) = im;
    end
else
    bl = mat2gray(bl);
end
end

function C = parity_add_remapping_drifting(C,opt)
Nneu2 = opt.Nneu2;
ses = opt.ses;
F = opt.F;
if ses>1
    temp = double(rand(Nneu2,ses)<=opt.overlap);
    t2 = repelem(eye(ses),ceil(Nneu2/ses),1);
    t2(randperm(size(t2,1),size(t2,1)-size(temp,1)),:) = [];
    temp = (temp+t2)>0;
    M = double(repelem(temp,1,F));
    M(M==0)=opt.overlapMulti;
else
    M=ones(Nneu2,F);
end
C = C.*M;
if opt.drift==1
    M = sim_drifting_activities(Nneu2,ses);
    M = double(repelem(M,1,F));
    C = C.*M;
end
end

function [dx,dy] = parity_make_xy_motion_ar(T, phi, std_px, max_px, seed)
if nargin<2, phi=0.95; end
if nargin<3, std_px=2.0; end
if nargin<4, max_px=[]; end
if nargin<5, rng('default'); else, rng(seed); end
sigma = std_px*sqrt(1-phi^2);
e1 = sigma*randn(1,T); e2 = sigma*randn(1,T);
dx = filter(1,[1 -phi], e1);
dy = filter(1,[1 -phi], e2);
dx = dx - mean(dx); dy = dy - mean(dy);
if ~isempty(max_px)
    s = max(1, max([abs(dx) abs(dy)],[],'all')/max_px);
    dx = dx./s; dy = dy./s;
end
end

function [dx,dy] = parity_make_xy_motion_ar_from_innov(phi, max_px, e1, e2)
dx = filter(1,[1 -phi], e1);
dy = filter(1,[1 -phi], e2);
dx = dx - mean(dx); dy = dy - mean(dy);
if ~isempty(max_px)
    s = max(1, max([abs(dx) abs(dy)],[],'all')/max_px);
    dx = dx./s; dy = dy./s;
end
end

function Yout = parity_apply_xy_translation(Y, dx, dy)
[~,~,T] = size(Y);
Yout = zeros(size(Y),'like',Y);
for k=1:T
    Yout(:,:,k) = imtranslate(Y(:,:,k), [dx(k) dy(k)], 'linear', ...
        'OutputView','same','FillValues',0);
end
end

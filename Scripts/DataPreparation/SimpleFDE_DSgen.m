% ============================================================
% Generate fractional forced benchmark datasets with FOMCON
%
% Creates three dataset groups:
%   1) reconstruction
%   2) extrapolation
%   3) forcing
%
% Folder structure:
%   base_dir/
%       reconstruction/train
%       reconstruction/val
%       reconstruction/test
%       extrapolation/train
%       extrapolation/val
%       extrapolation/test
%       forcing/train
%       forcing/val
%       forcing/test
%
% Each trajectory is saved as one CSV with columns:
%   t,u,y
%
% Requires:
%   - MATLAB
%   - FOMCON installed and on path
% ============================================================

clear;
clc;
close all;

%% -------------------------
% 0) Global settings
% --------------------------
script_dir = fileparts(mfilename('fullpath'));
base_dir = fullfile(fileparts(fileparts(script_dir)), 'data', 'Fractional_Data', 'fractional_benchmark_data');

% Number of trajectories per split
n_train = 40;
n_val   = 20;
n_test  = 20;

% Sampling
dt = 0.05;

% Fractional linear test system:
%   G(s) = b / (s^alpha + a)
alpha = 0.7;
a     = 1.0;
b     = 1.0;

s = fotf('s');
G = b / (s^alpha + a);

% Remove previously generated dataset folders to avoid mixing old and new files.
reset_generated_dataset_dirs(base_dir, {'reconstruction', 'extrapolation', 'forcing'});

% Recreate the folder structure expected by the Python datasets.
create_dataset_dir_tree(base_dir, {'reconstruction', 'extrapolation', 'forcing'});

%% -------------------------
% 1) Reconstruction dataset
% Same horizon, same forcing family across splits
% All trajectories start at t=0 without warmup
% --------------------------
disp('Generating reconstruction dataset ...');

T_rec = 20.0;

rng(42);
generate_split_reconstruction(G, dt, T_rec, n_train, ...
    fullfile(base_dir, 'reconstruction', 'train'), ...
    'train');

rng(43);
generate_split_reconstruction(G, dt, T_rec, n_val, ...
    fullfile(base_dir, 'reconstruction', 'val'), ...
    'val');

rng(44);
generate_split_reconstruction(G, dt, T_rec, n_test, ...
    fullfile(base_dir, 'reconstruction', 'test'), ...
    'test');

%% -------------------------
% 2) Extrapolation dataset
% Same forcing family, but different horizons
% --------------------------
disp('Generating extrapolation dataset ...');

T_train_extra = 6.0;
T_val_extra   = 12.0;
T_test_extra  = 20.0;

rng(52);
generate_split_extrapolation(G, dt, T_train_extra, n_train, ...
    fullfile(base_dir, 'extrapolation', 'train'), ...
    'train');

rng(53);
generate_split_extrapolation(G, dt, T_val_extra, n_val, ...
    fullfile(base_dir, 'extrapolation', 'val'), ...
    'val');

rng(54);
generate_split_extrapolation(G, dt, T_test_extra, n_test, ...
    fullfile(base_dir, 'extrapolation', 'test'), ...
    'test');

%% -------------------------
% 3) Forcing generalisation dataset
% Same horizon, but train/val/test use different forcing parameter ranges
% --------------------------
disp('Generating forcing dataset ...');

T_force = 20.0;

rng(62);
generate_split_forcing(G, dt, T_force, n_train, ...
    fullfile(base_dir, 'forcing', 'train'), ...
    'train');

rng(63);
generate_split_forcing(G, dt, T_force, n_val, ...
    fullfile(base_dir, 'forcing', 'val'), ...
    'val');

rng(64);
generate_split_forcing(G, dt, T_force, n_test, ...
    fullfile(base_dir, 'forcing', 'test'), ...
    'test');

disp('Done.');


% ============================================================
% Local helper functions
% ============================================================

function generate_split_reconstruction(G, dt, T_obs, n_samples, out_dir, split_name)
    % Reconstruction task:
    %   - same observation horizon for all splits
    %   - same forcing family for all splits
    %   - trajectories start directly at t=0
    %
    % This is an in-distribution full-trajectory task without a pre-history
    % warmup. With lsim(...) this means y starts from the default zero state.

    t = (0:dt:T_obs)';

    for i = 1:n_samples
        u = make_reconstruction_input(t, i);
        y = lsim(G, u, t);

        T = table(t, u, y);
        filename = fullfile(out_dir, sprintf('traj_%04d.csv', i));
        writetable(T, filename);
    end
end


function generate_split_extrapolation(G, dt, T_end, n_samples, out_dir, split_name)
    % Extrapolation task:
    %   - same smooth forcing family for train/val/test
    %   - only the horizon changes between splits
    %
    % This isolates horizon extrapolation from forcing-distribution shift.
    t = (0:dt:T_end)';

    for i = 1:n_samples
        u = make_extrapolation_input(t, i);
        y = lsim(G, u, t);

        T = table(t, u, y);
        filename = fullfile(out_dir, sprintf('traj_%04d.csv', i));
        writetable(T, filename);
    end
end


function generate_split_forcing(G, dt, T_end, n_samples, out_dir, split_name)
    % Special generator for unseen-forcing setup:
    % same horizon, but parameter ranges differ by split

    t = (0:dt:T_end)';

    for i = 1:n_samples
        u = make_forcing_generalisation_input(t, split_name, i);
        y = lsim(G, u, t);

        T = table(t, u, y);
        filename = fullfile(out_dir, sprintf('traj_%04d.csv', i));
        writetable(T, filename);
    end
end


function u = make_reconstruction_input(t, idx)
    % Smooth forcing family used for in-distribution reconstruction.
    %
    % Train/val/test are sampled from the same parameter ranges.
    % Different trajectories arise from different random draws and the warmup.

    amp1_range = [0.4, 0.9];
    amp2_range = [0.1, 0.4];
    w1_range   = [0.2, 0.6];
    w2_range   = [0.8, 1.4];
    c0_range   = [-0.1, 0.1];

    a1 = rand_range(amp1_range);
    a2 = rand_range(amp2_range);
    w1 = rand_range(w1_range);
    w2 = rand_range(w2_range);
    p1 = rand_range([0, 2*pi]);
    p2 = rand_range([0, 2*pi]);
    c0 = rand_range(c0_range);

    u = c0 + a1*sin(w1*t + p1) + a2*sin(w2*t + p2);
end


function u = make_extrapolation_input(t, idx)
    % Smooth input family:
    %   u(t) = c0 + a1 sin(w1 t + p1) + a2 sin(w2 t + p2)
    %
    % Same parameter ranges for all splits. Only horizon differs.

    amp1_range = [0.5, 0.8];
    amp2_range = [0.1, 0.3];
    w1_range   = [0.2, 0.5];
    w2_range   = [0.7, 1.2];
    c0_range   = [-0.1, 0.1];

    a1 = rand_range(amp1_range);
    a2 = rand_range(amp2_range);
    w1 = rand_range(w1_range);
    w2 = rand_range(w2_range);
    p1 = rand_range([0, 2*pi]);
    p2 = rand_range([0, 2*pi]);
    c0 = rand_range(c0_range);

    u = c0 + a1*sin(w1*t + p1) + a2*sin(w2*t + p2);
end


function u = make_forcing_generalisation_input(t, split_name, idx)
    % Train/val/test differ in forcing parameter ranges.
    % This creates an unseen-forcing setup.

    switch split_name
        case 'train'
            amp1_range = [0.4, 0.7];
            amp2_range = [0.1, 0.25];
            w1_range   = [0.20, 0.45];
            w2_range   = [0.80, 1.10];
            c0_range   = [-0.05, 0.05];

        case 'val'
            amp1_range = [0.5, 0.8];
            amp2_range = [0.15, 0.30];
            w1_range   = [0.25, 0.55];
            w2_range   = [0.90, 1.25];
            c0_range   = [-0.10, 0.10];

        case 'test'
            % deliberately shifted / broader
            amp1_range = [0.8, 1.1];
            amp2_range = [0.25, 0.45];
            w1_range   = [0.55, 0.85];
            w2_range   = [1.25, 1.70];
            c0_range   = [-0.15, 0.15];

        otherwise
            error('Unknown split_name.');
    end

    a1 = rand_range(amp1_range);
    a2 = rand_range(amp2_range);
    w1 = rand_range(w1_range);
    w2 = rand_range(w2_range);
    p1 = rand_range([0, 2*pi]);
    p2 = rand_range([0, 2*pi]);
    c0 = rand_range(c0_range);

    u = c0 + a1*sin(w1*t + p1) + a2*sin(w2*t + p2);
end


function x = rand_range(r)
    x = r(1) + (r(2) - r(1)) * rand();
end


function ensure_dir(p)
    if ~exist(p, 'dir')
        mkdir(p);
    end
end


function create_dataset_dir_tree(base_dir, dataset_names)
    ensure_dir(base_dir);
    splits = {'train', 'val', 'test'};
    for i = 1:numel(dataset_names)
        dataset_dir = fullfile(base_dir, dataset_names{i});
        ensure_dir(dataset_dir);
        for j = 1:numel(splits)
            ensure_dir(fullfile(dataset_dir, splits{j}));
        end
    end
end


function reset_generated_dataset_dirs(base_dir, dataset_names)
    for i = 1:numel(dataset_names)
        dataset_dir = fullfile(base_dir, dataset_names{i});
        if exist(dataset_dir, 'dir')
            fprintf('Removing old generated dataset folder: %s\n', dataset_dir);
            rmdir(dataset_dir, 's');
        end
    end
end

% RX_rop.m
% 功能：处理不同接收光功率 (ROP) 的 VPI 仿真数据，生成 Python 训练/测试用的 .mat 文件。
%
% 输入文件命名约定（放在 ROP测试数据\ 子文件夹下）:
%   vpi_data_5km_bw25_laser0.5_rop0.txt    → ROP = 0 dBm
%   vpi_data_5km_bw25_laser0.5_rop-1.txt   → ROP = -1 dBm
%   ...
%   vpi_data_5km_bw25_laser0.5_rop-15.txt  → ROP = -15 dBm
%
% 输出:
%   dataset_rop0_for_python.mat          ← ROP=0 的训练+测试数据（根目录）
%   ROP测试数据/dataset_rop0_test.mat    ← 各 ROP 的测试数据（含 ROP=0）
%   ROP测试数据/dataset_rop-1_test.mat
%   ...
%   ROP测试数据/dataset_rop-15_test.mat

config;

rop_dir = fullfile(pwd, 'ROP测试数据');
if ~exist(rop_dir, 'dir'), mkdir(rop_dir); end

symb_train = load('symb_train.txt');
symb_test  = load('symb_test.txt');
symb_train_export = symb_train(:);
symb_test_export  = symb_test(:);

rop_list = 0:-1:-15;

for idx = 1:numel(rop_list)
    rop   = rop_list(idx);
    fname = fullfile(rop_dir, sprintf('vpi_data_5km_bw25_laser0.5_rop%d.txt', rop));

    if ~isfile(fname)
        fprintf('[跳过] 文件不存在: %s\n', fname);
        continue;
    end

    %% 信号预处理（与 RX.m 保持一致）
    rx = -load(fname);
    rx = 2*(rx - mean(rx)) / mean(abs(rx));

    rx_train = rx(1:nSymbols_train*sps);
    rx_test  = rx(nSymbols_test*sps + 1:end);

    rx_matched_train = conv(rx_train, rrc, 'same');
    rx_matched_test  = conv(rx_test,  rrc, 'same');

    rx_train_export = rx_matched_train(:);
    rx_test_export  = rx_matched_test(:);

    %% ROP=0: 同时生成训练+测试 mat（供 train_all_rop.py 使用）
    if rop == 0
        save('dataset_rop0_for_python.mat', ...
             'rx_train_export', 'rx_test_export', ...
             'symb_train_export', 'symb_test_export', 'sps');
        fprintf('[OK] ROP=0 训练+测试 → dataset_rop0_for_python.mat\n');
    end

    %% 所有 ROP 均生成测试专用 mat（供 test_rop.py 使用）
    mat_name = fullfile(rop_dir, sprintf('dataset_rop%d_test.mat', rop));
    save(mat_name, 'rx_test_export', 'symb_test_export', 'sps');
    fprintf('[OK] ROP=%d dBm → %s\n', rop, mat_name);
end

fprintf('\n全部 ROP 数据处理完成。\n');

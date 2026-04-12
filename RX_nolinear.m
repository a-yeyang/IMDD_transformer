% RX_nolinear.m — 批量处理非线性效应 VPI 仿真数据
%
% 遍历 "非线性效应数据" 文件夹下的全部 VPI 仿真文件，
% 对每组数据执行 RRC 匹配滤波后，保存为供 Python 读取的 .mat 文件。
%
% 实验条件:
%   - BTB (Back-to-Back, 无光纤): 激光器驱动电压 0.5/0.75/1.0/1.25/1.5 V
%   - 5km SSMF 光纤:             激光器驱动电压 0.5/0.75/1.0/1.25/1.5 V
%   驱动电压越大，调制器非线性效应越强
%
% 输出目录: nolinear_results/mat_data/
%   共生成 10 个 .mat 文件，Python 脚本直接读取

config;   % 加载公共参数 (Fs, Rs, sps, rrc, nSymbols_train, nSymbols_test 等)

% ---- 路径设置 ----
data_dir   = [fileparts(mfilename('fullpath')), '\非线性效应数据\'];
output_dir = [fileparts(mfilename('fullpath')), '\nolinear_results\mat_data\'];

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('[INFO] 创建输出目录: %s\n', output_dir);
end

% ---- 数据集定义 {VPI文件名, 输出条件标识} ----
datasets = {
    'vpi_data_BTB_laser0.5.txt',   'BTB_laser0.5';
    'vpi_data_BTB_laser0.75.txt',  'BTB_laser0.75';
    'vpi_data_BTB_laser1.0.txt',   'BTB_laser1.0';
    'vpi_data_BTB_laser1.25.txt',  'BTB_laser1.25';
    'vpi_data_BTB_laser1.5.txt',   'BTB_laser1.5';
    'vpi_data_5km_laser0.5.txt',   '5km_laser0.5';
    'vpi_data_5km_laser0.75.txt',  '5km_laser0.75';
    'vpi_data_5km_laser1.txt',     '5km_laser1.0';
    'vpi_data_5km_laser1.25.txt',  '5km_laser1.25';
    'vpi_data_5km_laser1.5.txt',   '5km_laser1.5';
};

n_conditions = size(datasets, 1);

% ---- 加载公共符号标签 (所有条件使用相同的发送序列) ----
root_dir = fileparts(mfilename('fullpath'));
symb_train_raw = load([root_dir, '\symb_train.txt']);
symb_test_raw  = load([root_dir, '\symb_test.txt']);
symb_train_export = symb_train_raw(:);
symb_test_export  = symb_test_raw(:);

fprintf('\n========================================\n');
fprintf(' RX_nolinear.m — 批量 VPI 数据预处理\n');
fprintf('========================================\n');
fprintf(' 训练符号数: %d\n', length(symb_train_export));
fprintf(' 测试符号数: %d\n', length(symb_test_export));
fprintf(' 采样率 Fs = %.1f GHz, 符号率 Rs = %.1f GBaud\n', Fs, Rs);
fprintf(' 当前 sps = %d (每符号采样点数)\n', sps);
fprintf(' 将处理 %d 组条件数据\n\n', n_conditions);

% ---- 批量处理 ----
for i = 1:n_conditions
    vpi_filename = datasets{i, 1};
    cond_id      = datasets{i, 2};
    vpi_filepath = [data_dir, vpi_filename];

    fprintf('[%2d/%2d] 处理条件: %-20s <- %s\n', i, n_conditions, cond_id, vpi_filename);

    % 检查文件是否存在
    if ~exist(vpi_filepath, 'file')
        fprintf('         [警告] 文件不存在，跳过: %s\n', vpi_filepath);
        continue;
    end

    % 加载 VPI 原始信号并预处理（取负值 + 幅度归一化）
    rx_raw = -load(vpi_filepath);
    rx_raw = 2 * (rx_raw - mean(rx_raw)) / mean(abs(rx_raw));
    rx_raw = rx_raw(:);

    total_samples = length(rx_raw);
    expected_samples = (nSymbols_train + nSymbols_test) * sps;
    if total_samples < expected_samples
        fprintf('         [警告] 信号长度 %d < 期望 %d，截断处理\n', ...
                total_samples, expected_samples);
    end

    % 分割训练集和测试集
    rx_train_raw = rx_raw(1 : nSymbols_train * sps);
    rx_test_raw  = rx_raw(nSymbols_test * sps + 1 : min(end, (nSymbols_train + nSymbols_test) * sps));

    % RRC 匹配滤波（conv 'same' 保持长度不变）
    rx_matched_train = conv(rx_train_raw, rrc, 'same');
    rx_matched_test  = conv(rx_test_raw,  rrc, 'same');

    % 转列向量导出
    rx_train_export = rx_matched_train(:);
    rx_test_export  = rx_matched_test(:);

    % 保存 .mat（sps 保持 config 中的值，Python 端同步使用）
    out_file = [output_dir, 'dataset_', cond_id, '.mat'];
    save(out_file, ...
         'rx_train_export', 'rx_test_export', ...
         'symb_train_export', 'symb_test_export', 'sps');

    fprintf('         已保存 -> %s\n', out_file);
    fprintf('         训练集: %d 样本 | 测试集: %d 样本\n\n', ...
            length(rx_train_export), length(rx_test_export));
end

fprintf('========================================\n');
fprintf(' 全部 %d 组数据预处理完成！\n', n_conditions);
fprintf(' 输出目录: %s\n', output_dir);
fprintf('========================================\n');

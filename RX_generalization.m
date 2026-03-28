% RX_generalization.m — 批量处理「泛化测试数据」目录下多组 VPI 输出，生成供 Python 使用的 .mat
%
% 用法：在工程根目录运行本脚本。依赖 utils/config.m、根目录的 symb_train.txt / symb_test.txt。
% 输出：泛化测试数据/dataset_for_python_bw<带宽>.mat（与 vpi 文件名中的 bw 一致）

script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'utils'));
config;

data_dir = fullfile(script_dir, '泛化测试数据');
if ~isfolder(data_dir)
    error('未找到目录: %s', data_dir);
end

symb_train = load(fullfile(script_dir, 'symb_train.txt'));
symb_test  = load(fullfile(script_dir, 'symb_test.txt'));

flist = dir(fullfile(data_dir, 'vpi_data*.txt'));
if isempty(flist)
    error('在 %s 下未找到 vpi_data*.txt 文件', data_dir);
end

fprintf('共 %d 个 VPI 文件待处理。\n', numel(flist));

for k = 1:numel(flist)
    fname = flist(k).name;
    tok = regexp(fname, 'bw([\d.]+)', 'tokens', 'once');
    if isempty(tok)
        warning('跳过无法解析带宽的文件: %s', fname);
        continue
    end
    bw_str = tok{1};
    bandwidth_ghz = str2double(bw_str);

    fpath = fullfile(data_dir, fname);
    fprintf('处理 [%s]  bandwidth = %s GHz ...\n', fname, bw_str);

    rx = -load(fpath);
    rx = 2 * (rx - mean(rx)) / mean(abs(rx));

    rx_train = rx(1:nSymbols_train * sps);
    rx_test  = rx(nSymbols_test * sps + 1:end);

    rx_matched_train = conv(rx_train, rrc, 'same');
    rx_matched_test  = conv(rx_test,  rrc, 'same');

    sps = 2;
    rx_sym_train = resample(rx_matched_train, Rs * sps, Fs)';
    rx_sym_test  = resample(rx_matched_test,  Rs * sps, Fs)';

    rx_train_export = rx_matched_train(:);
    rx_test_export  = rx_matched_test(:);

    symb_train_export = symb_train(:);
    symb_test_export  = symb_test(:);

    source_vpi_file = fname;

    out_name = sprintf('dataset_for_python_bw%s.mat', bw_str);
    out_path = fullfile(data_dir, out_name);

    save(out_path, ...
        'rx_train_export', 'rx_test_export', ...
        'symb_train_export', 'symb_test_export', ...
        'sps', 'bandwidth_ghz', 'source_vpi_file');

    fprintf('  已保存: %s\n', out_name);
end

fprintf('全部完成。数据目录: %s\n', data_dir);

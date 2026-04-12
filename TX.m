%  发射机程序
% （RRC成形、简化信道、匹配滤波/抽样）。
clear; close all; clc;
%% ----------------- 仿真/网络 超参数（保持原设定，除模型外不改） -----------------
useGPU = true;            % 是否使用 GPU（自动检测）；若你的 MATLAB/平台 不支持 GPU，可设 false
rngSeed = 1;          % 随机数种子（使用 Mersenne Twister）
rng(rngSeed, 'twister');

config;

% PAM4 符号列表（{-3,-1,1,3}）
pam4_levels = [-3, -1, 1, 3];

%% ----------------- 产生数据（发射端） -----------------
% 生成训练 & 测试符号（二进制 → Gray 映射 → PAM4）
numTrain = nSymbols_train;
numTest  = nSymbols_test;
totalSymbols = numTrain + numTest;

if enable_RS
    fprintf('\n========== 启用RS码编码 ==========\n');
    fprintf('RS参数: RS(%d,%d) over GF(2^%d)\n', n_rs, k_rs, m_rs);
    fprintf('码率: %.4f (%.2f%%)\n', k_rs/n_rs, k_rs/n_rs*100);
    fprintf('纠错能力: %d个符号错误\n\n', t_rs);
    
    % 计算需要的RS码字数量
    % totalSymbols个PAM4符号 -> totalSymbols/4个GF(2^8)符号 -> (totalSymbols/4)/n_rs个码字
    numGF_symbols = totalSymbols / 4;  % 每4个PAM4符号(8bit)对应1个GF(2^8)符号
    numCodewords = numGF_symbols / n_rs;  % 每个码字有n_rs个编码符号
    
    if mod(numCodewords, 1) ~= 0
        error('符号数量%d无法被RS码字长度%d整除！', totalSymbols, n_rs*4);
    end
    
    fprintf('生成 %d 个RS码字（每码字%d个信息符号）...\n', numCodewords, k_rs);
    
    % 1. 生成原始信息符号 (GF(2^m)符号，范围0到2^m-1)
    data_info = randi([0, 2^m_rs-1], numCodewords, k_rs);  % [numCodewords, k_rs]
    
    % 2. RS编码
    msg_gf = gf(data_info, m_rs);                        % [numCodewords, k_rs]
    encoded_gf = rsenc(msg_gf, n_rs, k_rs).';            % [numCodewords, n_rs]
    
    % 3. 转换为比特流
    encoded_bits = de2bi(double(encoded_gf.x), m_rs, 'left-msb');  % [numCodewords*n_rs, m_rs]
    encoded_bits_1d = reshape(encoded_bits.', [], 1);    % [numCodewords*n_rs*m_rs, 1]
    
    % 4. PAM4调制：每2个比特映射到一个符号
    % Gray映射: 00->-3, 01->-1, 11->1, 10->3
    bits = encoded_bits_1d;
    symb = zeros(length(bits)/2, 1);
    for i = 1:length(symb)
        b1 = bits(2*i-1);
        b2 = bits(2*i);
        if b1==0 && b2==0
            symb(i) = -3;
        elseif b1==0 && b2==1
            symb(i) = -1;
        elseif b1==1 && b2==1
            symb(i) = 1;
        else  % b1==1 && b2==0
            symb(i) = 3;
        end
    end
    
    % 保存原始信息数据用于接收端解码后比较
    save('rs_info_data.mat', 'data_info', 'numCodewords');
    fprintf('原始信息数据已保存到 rs_info_data.mat\n');
    
else
    fprintf('\n========== 不使用RS码 ==========\n');
    
    % 原始方法：直接生成PAM4符号
    % 2 bits per symbol
    bits = randi([0 1], 2*totalSymbols, 1, 'uint8');  
    % Gray mapping: 00->-3, 01->-1, 11->1, 10->3
    mapGray = containers.Map({'00','01','11','10'}, {-3,-1,1,3});

    symb = zeros(totalSymbols,1);
    for i=1:totalSymbols
        b1 = num2str(bits(2*i-1));
        b2 = num2str(bits(2*i));
        key = [b1 b2];
        symb(i) = mapGray(key);
    end
end

fprintf('总共生成 %d 个PAM4符号\n', length(symb));

% 划分训练/测试
symb_train = symb(1:numTrain);
symb_test  = symb(numTrain+1:end);

% 上采样 & RRC 成形（发射）
tx_up_train = upsample(symb_train, sps);
tx_up_test  = upsample(symb_test, sps);
tx_train = conv(tx_up_train, rrc, 'same');
tx_test  = conv(tx_up_test,  rrc, 'same');



% 定义期望的3dB带宽（使用config.m中的bandwidth参数）
% 巴特沃斯滤波器阶数
N = 1;  % 可以调节，阶数越高过渡带越陡

% 定义期望的3dB带宽（使用config.m中的bandwidth参数）
B_3dB = bandwidth; % GHz

% 修正：filtfilt双向滤波的截止频率补偿
% filtfilt使得幅频响应平方，因此要求 (1 + (B_3dB/fc)^(2N))^-2 = 0.5
% 解得正确的补偿系数
compensation_factor = (sqrt(2) - 1)^(-1 / (2*N));
B = B_3dB * compensation_factor; % 补偿后的实际截止频率 (单位为GHz，与Fs保持一致)

% 修正：Wn必须除以奈奎斯特频率 (Fs / 2)，而不是符号率 Rs
Wn = B / (Fs / 2);

% 安全性限制：防止设定的带宽补偿后超过奈奎斯特频率导致报错
if Wn >= 1
    warning('补偿后的截止频率超出了奈奎斯特频率限制，将Wn限制为0.99');
    Wn = 0.99;
end

% 设计低通巴特沃斯滤波器
[b,a] = butter(N, Wn, 'low');

% 进行双向滤波
tx_train_bl = filtfilt(b,a, tx_train);
tx_test_bl  = filtfilt(b,a, tx_test);

% 查看单次滤波器的频响特性（注意：fvtool显示的是单次滤波的响应，-3dB点在补偿后的 Wn 处）
fvtool(b,a)

if (bw_limit~=1.1)
    tx_all = [tx_train_bl; tx_test_bl]';
else
tx_all = [tx_train; tx_test]';
end

tx_all=tx_all*TX_k;
symb_all = [symb_train; symb_test]';
save('tx_all_to_vpi.txt', 'tx_all','-ascii');
save('symb_all.txt', 'symb_all','-ascii');
save('symb_train.txt','symb_train','-ascii');
save('symb_test.txt','symb_test','-ascii');

%% 频谱图选项（采样率 Fs 与 Plotspectrum 频率轴一致）
% plotSpectrumBwOnly：true=只画带限后（红线）；false=无带限与带限对比（蓝+红）
% plotSpectrumShowLegend / plotSpectrumShowYLabel：是否显示图例、Y 轴 “Magnitude (dB)” 标签
plotSpectrumBwOnly = true;
plotSpectrumShowLegend = false;
plotSpectrumShowYLabel = true;

specOpts = {'showlegend', plotSpectrumShowLegend, 'showylabel', plotSpectrumShowYLabel};

if plotSpectrumBwOnly
    % 仅带限后频谱（双边谱、单边谱各一图）
    Plotspectrum(tx_train_bl, Fs, false, 'color', 'r', 'label', 'Bandwidth Limit', specOpts{:});
    Plotspectrum(tx_train_bl, Fs, true, 'color', 'r', 'label', 'After Butterworth', specOpts{:});
else
    Plotspectrum(tx_train, Fs, false, 'label', 'No Bandwidth Limit', 'color', 'b', specOpts{:});
    Plotspectrum(tx_train_bl, Fs, true, 'newfig', true, 'label', 'Bandwidth Limit', 'color', 'r', specOpts{:});
    Plotspectrum(tx_train, Fs, true, 'label', 'Before filtering', 'color', 'b', specOpts{:});
    Plotspectrum(tx_train_bl, Fs, true, 'newfig', true, 'label', 'After Butterworth', 'color', 'r', specOpts{:});
end
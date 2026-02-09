%  发射机程序
% （RRC成形、简化信道、匹配滤波/抽样）。
clear; close all; clc;
%% ----------------- 仿真/网络 超参数（保持原设定，除模型外不改） -----------------
useGPU = true;            % 是否使用 GPU（自动检测）；若你的 MATLAB/平台 不支持 GPU，可设 false
rngSeed = 11;          % 随机数种子（使用 Mersenne Twister）
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

% 巴特沃斯滤波器阶数
N = 1;  % 可以调节，阶数越高过渡带越陡

% 定义期望的3dB带宽（使用config.m中的bandwidth参数）
B_3dB = bandwidth; % GHz

% 由于使用filtfilt进行双向滤波，需要补偿截止频率
% 使得最终的3dB带宽为B_3dB
% 补偿公式: fc = B_3dB * (2^(1/N) - 1)^(-1/2)
compensation_factor = (2^(1/N) - 1)^(-0.5);
B = B_3dB * compensation_factor; % Hz, 补偿后的截止频率

%
Wn = B / Rs;

% 设计低通巴特沃斯滤波器
[b,a] = butter(N, Wn, 'low');
tx_train_bl = filtfilt(b,a, tx_train);
tx_test_bl  = filtfilt(b,a, tx_test);
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


% 原始信号和带限后的信号
sig1 = tx_train;        % 带限前
sig2 = tx_train_bl;   % 带限后

Nfft = 2^14;            % FFT 点数，取大一点看得更细
f = (-Nfft/2:Nfft/2-1)*(Fs/Nfft);  % 频率坐标 (Hz)

% 归一化幅度谱（双边谱）
S1 = fftshift(abs(fft(sig1, Nfft)))/max(abs(fft(sig1, Nfft)));
S2 = fftshift(abs(fft(sig2, Nfft)))/max(abs(fft(sig2, Nfft)));

% 画图1 - 双边谱（与plotspectrum.m风格一致）
figure('Color', 'white', 'Position', [100, 100, 800, 600]);  % 白色背景，设置图形大小
plot(f/1e9, 20*log10(S1+eps), 'b-', 'LineWidth', 1.5); hold on;
plot(f/1e9, 20*log10(S2+eps), 'r-', 'LineWidth', 1.5);

% 设置坐标轴标签 - 与plotspectrum.m风格一致
xlabel('Frequency (GHz)', 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
ylabel('Magnitude (dB)', 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Times New Roman');

% 设置图例 - 与plotspectrum.m风格一致
legend('No Bandwidth Limit', 'Bandwidth Limit', 'FontSize', 18, 'FontWeight', 'bold', ...
       'Location', 'best', 'Box', 'on', 'EdgeColor', 'black', 'LineWidth', 1.0, 'FontName', 'Times New Roman');

% 设置坐标轴属性 - 与plotspectrum.m风格一致
ax = gca;
ax.FontSize = 18;  % 坐标轴刻度字体大小
ax.FontWeight = 'bold';
ax.FontName = 'Times New Roman';
ax.LineWidth = 1;  % 坐标轴线宽
ax.Box = 'on';  % 显示坐标轴框
ax.GridLineStyle = '-';  % 主网格线样式（实线）
ax.GridAlpha = 0.3;  % 主网格线透明度
ax.MinorGridLineStyle = ':';  % 次网格线样式（点线）
ax.MinorGridAlpha = 0.2;  % 次网格线透明度
ax.TickLength = [0.01, 0.02];  % 刻度长度

grid on;
grid minor;
axis tight;

% 单边谱计算（只取正频率部分）
f_single = (0:Nfft/2-1)*(Fs/Nfft);  % 单边频率坐标 (0 到 Fs/2)
S1_single = abs(fft(sig1, Nfft));   % 双边FFT结果
S2_single = abs(fft(sig2, Nfft));   % 双边FFT结果
S1_single = S1_single(1:Nfft/2);   % 只取正频率部分
S2_single = S2_single(1:Nfft/2);   % 只取正频率部分
S1_single = S1_single / max(S1_single);  % 归一化
S2_single = S2_single / max(S2_single);  % 归一化

% 降采样以获得清晰的轮廓线（每隔N个点取一个）
downsample_factor = max(1, floor(length(f_single) / 1000));  % 大约保留1000个点
idx_downsample = 1:downsample_factor:length(f_single);
f_single_outline = f_single(idx_downsample);
S1_single_outline = S1_single(idx_downsample);
S2_single_outline = S2_single(idx_downsample);

% 画图2 - 单边谱（与plotspectrum.m风格一致）
figure('Color', 'white', 'Position', [100, 100, 800, 600]);  % 白色背景，设置图形大小
plot(f_single_outline/1e9, 20*log10(S1_single_outline+eps), 'b-', 'LineWidth', 1.5, 'Marker', 'none'); hold on;
plot(f_single_outline/1e9, 20*log10(S2_single_outline+eps), 'r-', 'LineWidth', 1.5, 'Marker', 'none');

% 设置坐标轴标签 - 与plotspectrum.m风格一致
xlabel('Frequency (GHz)', 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
ylabel('Magnitude (dB)', 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Times New Roman');

% 设置图例 - 与plotspectrum.m风格一致
legend('Before filtering', 'After Butterworth', 'FontSize', 18, 'FontWeight', 'bold', ...
       'Location', 'best', 'Box', 'on', 'EdgeColor', 'black', 'LineWidth', 1.0, 'FontName', 'Times New Roman');

% 设置坐标轴属性 - 与plotspectrum.m风格一致
ax = gca;
ax.FontSize = 18;  % 坐标轴刻度字体大小
ax.FontWeight = 'bold';
ax.FontName = 'Times New Roman';
ax.LineWidth = 1;  % 坐标轴线宽
ax.Box = 'on';  % 显示坐标轴框
ax.GridLineStyle = '-';  % 主网格线样式（实线）
ax.GridAlpha = 0.3;  % 主网格线透明度
ax.MinorGridLineStyle = ':';  % 次网格线样式（点线）
ax.MinorGridAlpha = 0.2;  % 次网格线透明度
ax.TickLength = [0.01, 0.02];  % 刻度长度

grid on;
grid minor;
axis tight;
xlim([0, max(f_single_outline/1e9)]);  % 只显示正频率部分
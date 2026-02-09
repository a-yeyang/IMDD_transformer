config;
rx=-load('vpi_data.txt');
rx=2*(rx-mean(rx))/mean(abs(rx));
%% ===================================================
rx_train=rx(1:nSymbols_train*sps);
rx_test=rx(nSymbols_test*sps+1:end);
%% ----------------- 匹配滤波+下采样 ----------------- 
rx_matched_train = conv(rx_train, rrc,'same');
rx_matched_test  = conv(rx_test,  rrc,'same');
sps=2;
rx_sym_train = resample(rx_matched_train,Rs*sps,Fs)';
rx_sym_test  = resample(rx_matched_test,Rs*sps,Fs)';
symb_train=load('symb_train.txt');

% --- 在你的接收机代码末尾添加 ---
% 假设 rx_matched_train 是经过RRC匹配滤波后的信号 (通常是2 SPS或更高)
% 我们需要保留过采样信号用于非线性补偿

% 确保数据是列向量
rx_train_export = rx_matched_train(:); 
rx_test_export  = rx_matched_test(:);

% 对应的标签 (Ground Truth)
symb_train_export = symb_train(:);
symb_test_export  = symb_test(:);

% 保存为 .mat 文件 (Python scipy.io.loadmat 读取更方便且不丢失精度)
save('dataset_for_python.mat', 'rx_train_export', 'rx_test_export', ...
     'symb_train_export', 'symb_test_export', 'sps');

fprintf('数据已保存供Python使用：rx (2 SPS), labels (Symbols)\n');
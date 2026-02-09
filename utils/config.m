global span rolloff Fs Rs GBt nSymbols_train nSymbols_test nSymbols_all ...
sps rrc UXR TX_k bw_limit SSMF_length laser_driver_amp enable_RS m_rs n_rs k_rs t_rs bandwidth; 

span  = 10;         % Raised cosine (combined Tx/Rx) delay 
rolloff  = 0.1;        % Raised cosine roll-off factor (0.1-0.5)
GBt=Rs;
UXR=256;



SSMF_length = 2;
TX_k=1;
laser_driver_amp=0.5;
Fs=120; 
Rs=60;
UXR=256;



% bandwidth: 期望的3dB带宽（单位：GHz）
% 注意：TX_data_generator.m中使用filtfilt双向滤波，已进行截止频率补偿
bandwidth=25;  % 最终信号的3dB带宽为30GHz
bw_limit=bandwidth/(Rs/2);  % 归一化带宽（保留用于兼容性）

sps=Fs/Rs;
nSymbols_train= 131072;
nSymbols_test=131072;
nSymbols_all=nSymbols_train+nSymbols_test;
rrc = rcosdesign(rolloff, span, sps, 'sqrt');

%% ========== RS码参数设置 ==========
enable_RS = false;     % RS码使能标志：true=启用RS码，false=不使用RS码
m_rs = 8;              % 每个RS符号包含的比特数（GF(2^8)=GF(256)）
n_rs = 128;            % RS码字长度（编码后符号数）
k_rs = 120;            % RS信息符号数（编码前符号数）
t_rs = (n_rs - k_rs) / 2;  % 纠错能力：最多纠正4个符号错误

% 说明：
% - 131072个PAM4符号 = 32768个GF(2^8)符号（因为8bit/4 = 2 PAM4符号）
% - 32768个GF(2^8)符号 / 128 = 256个RS码字（完美整除）
% - 每个码字：120个信息符号 -> 128个编码符号
% - 码率：R = 120/128 = 0.9375 (93.75%)




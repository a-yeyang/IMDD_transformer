%% BER vs SNR 折线图 (60G 带限20G, 5km训练, 2km~6km测试)
% 场景：光纤通信 IMDD PAM4
% 使用 semilogy 绘制误码率（对数坐标）
% 论文级美化，白底

clear; clc; close all;

%% 信噪比 (dB)：0, 5, 10, 15, 20, 无噪声用 25 表示（与前面等距），刻度显示为 Inf
SNR_dB = [0, 5, 10, 15, 20, 25];
% 用于对数坐标：BER=0 替换为最小显示值，避免 log(0)
BER_min = 1e-10;

%% 数据：每行为 [FCNN, DNN, BiLSTM, Transformer TEQ, KAN-Former]，列对应 SNR_dB
% 5km 测试
BER_5km = [
    0.58238, 0.46838, 0.29344, 0.10797, 0.014413, BER_min;   % FCNN (无噪声 0->BER_min)
    0.57231, 0.44451, 0.25951, 0.084678, 0.010972, 3.052e-5;
    0.57498, 0.47301, 0.31413, 0.12586, 0.018114, BER_min;
    0.53306, 0.39785, 0.22094, 0.075308, 0.017503, 0.0019533;
    0.53848, 0.39152, 0.21121, 0.061887, 0.0079352, 0.00016023
]';
% 2km 测试
BER_2km = [
    0.57160, 0.45475, 0.29539, 0.14892, 0.069159, 0.028292;
    0.56104, 0.42996, 0.25693, 0.10928, 0.039577, 0.012330;
    0.56717, 0.46104, 0.31020, 0.15912, 0.070143, 0.024393;
    0.52489, 0.38115, 0.19550, 0.055081, 0.0084846, 0.00051884;
    0.52489, 0.37414, 0.19267, 0.052754, 0.0080115, 0.00031283
]';
% 3km 测试
BER_3km = [
    0.57241, 0.45383, 0.28902, 0.13310, 0.049252, 0.011666;
    0.56129, 0.42940, 0.25137, 0.096253, 0.026110, 0.0037463;
    0.56671, 0.45931, 0.30383, 0.14386, 0.051587, 0.0094612;
    0.52479, 0.38018, 0.19199, 0.050274, 0.0060964, 7.63e-6;
    0.52737, 0.37294, 0.18569, 0.044079, 0.0033725, BER_min
]';
% 4km 测试
BER_4km = [
    0.57633, 0.45681, 0.28394, 0.11006, 0.021021, 0.00023653;
    0.56504, 0.43387, 0.24669, 0.079360, 0.010201, 3.052e-5;
    0.56905, 0.46256, 0.30165, 0.12599, 0.025866, 6.867e-5;
    0.52733, 0.38548, 0.19789, 0.053784, 0.0067831, 6.104e-5;
    0.53226, 0.37865, 0.19058, 0.044384, 0.0025484, BER_min
]';
% 6km 测试
BER_6km = [
    0.59404, 0.48797, 0.33315, 0.17342, 0.074057, 0.020235;
    0.58346, 0.46768, 0.30606, 0.15696, 0.077514, 0.037051;
    0.58489, 0.49041, 0.35029, 0.18515, 0.074912, 0.017602;
    0.54368, 0.41973, 0.26414, 0.13755, 0.075354, 0.044239;
    0.54946, 0.41520, 0.25537, 0.12099, 0.053982, 0.022867
]';

%% 模型名称与线型
modelNames = {'FCNN', 'DNN', 'BiLSTM', 'Transformer TEQ', 'KAN-Former'};
% 论文常用线型与符号
lineStyles = {'-o', '-s', '-d', '-^', '-v'};
% 颜色 (区分度高、适合打印)
colors = [
    0.00, 0.45, 0.74;   % 蓝
    0.85, 0.33, 0.10;   % 橙
    0.47, 0.67, 0.19;   % 绿
    0.49, 0.18, 0.56;   % 紫
    0.93, 0.69, 0.13    % 金
];

%% 子图数据与标题
datas = {BER_2km, BER_3km, BER_4km, BER_5km, BER_6km};
titles = {'测试 2 km', '测试 3 km', '测试 4 km', '测试 5 km', '测试 6 km'};

%% 创建图形：白底、论文级
fig = figure('Color', 'w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);
fig.PaperPositionMode = 'auto';
fig.PaperSize = [12 10];

% 字体：中文用宋体，英文/数字用新罗马 (Times New Roman)，全部加粗
fontNameEn = 'Times New Roman';   % 英文与数字
fontNameCn = 'SimSun';            % 宋体，中文
axLineWidth = 1.5;               % 坐标轴线宽（加粗）

legendHandles = [];  % 用于图例的线条句柄（从第一个子图获取）
for sp = 1:5
    subplot(2, 3, sp);
    hold on; grid on; box on;
    set(gca, 'Color', 'w', 'XColor', [0.2 0.2 0.2], 'YColor', [0.2 0.2 0.2], ...
        'LineWidth', axLineWidth, 'FontName', fontNameEn, 'FontSize', 14, 'FontWeight', 'bold');

    B = datas{sp};  % size: (6 x 5)，行=SNR，列=模型
    for m = 1:5
        ber = B(:, m);
        ber(ber <= 0) = BER_min;
        h = plot(SNR_dB, ber, lineStyles{m}, 'Color', colors(m,:), ...
            'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerFaceColor', colors(m,:));
        if sp == 1
            legendHandles(m) = h;
        end
    end

    set(gca, 'YScale', 'log');
    xlabel('SNR (dB)', 'FontName', fontNameEn, 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('BER', 'FontName', fontNameEn, 'FontSize', 14, 'FontWeight', 'bold');
    title(titles{sp}, 'FontName', fontNameCn, 'FontSize', 14, 'FontWeight', 'bold');
    xlim([0 25]);
    ylim([1e-4 1]);
    xticks([0 5 10 15 20 25]);
    xticklabels({'0', '5', '10', '15', '20', 'Inf'});
    hold off;
end

% 第六个子图位置放图例（使用第一个子图的线条句柄）
subplot(2, 3, 6);
axis off;
set(gca, 'Color', 'w');
legend(legendHandles, modelNames, 'Location', 'best', ...
    'FontName', fontNameEn, 'FontSize', 15, 'FontWeight', 'bold');

%% 总标题
sgtitle('60G带限20G,5km训练 不同传输距离下BER/SNR (PAM4 硬判决)', ...
    'FontName', fontNameCn, 'FontSize', 18, 'FontWeight', 'bold');

%% 保存
outDir = fileparts(mfilename('fullpath'));
if isempty(outDir), outDir = pwd; end
saveas(fig, fullfile(outDir, 'BER_vs_SNR_multidistance.fig'));
print(fig, fullfile(outDir, 'BER_vs_SNR_multidistance.png'), '-dpng', '-r300');
print(fig, fullfile(outDir, 'BER_vs_SNR_multidistance.pdf'), '-dpdf', '-vector');
fprintf('图形已保存至: %s\n', outDir);

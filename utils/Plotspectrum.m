%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  200Gb/s Project  Li Zhipei Wang Xishuo Yuan Gao 2021.8 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Plotspectrum(signal,baudrate,onesided,varargin)
% Plotspectrum(signal,baudrate) - 绘制信号频谱（双边谱）
% Plotspectrum(signal,baudrate,true) - 绘制单边谱
% Plotspectrum(signal,baudrate,onesided,'newfig',false,'hold',true) - 在同一图中绘制多个信号
% 输入参数:
%   signal   - 输入信号
%   baudrate - 波特率
%   onesided - 是否绘制单边谱 (可选，默认: false，即双边谱)
% 可选参数:
%   'newfig' - 是否创建新图形窗口 (默认: true)
%   'hold'   - 是否保持当前图形 (默认: false, 当newfig=false时自动为true)
%   'color'  - 线条颜色 (默认: 'b')
%   'label'  - 图例标签 (默认: '')

% 处理第三个参数（onesided）- 智能判断是否为逻辑值或可选参数
onesided_flag = false;  % 默认双边谱
if nargin >= 3
    % 判断第三个参数是否为逻辑值或数值（0/1）
    if islogical(onesided) || (isnumeric(onesided) && (onesided == 0 || onesided == 1))
        onesided_flag = logical(onesided); 
    else
        % 如果第三个参数不是逻辑值，将其作为可选参数处理
        varargin = [{onesided}, varargin];
    end
end

% 解析可选参数
p = inputParser;
addParameter(p,'newfig',true,@islogical);
addParameter(p,'hold',false,@islogical);
addParameter(p,'color','b',@(x) ischar(x) || isstring(x));
addParameter(p,'label','',@(x) ischar(x) || isstring(x));
parse(p,varargin{:});

newfig = p.Results.newfig;
holdon = p.Results.hold;
linecolor = p.Results.color;
linelabel = p.Results.label;

% 如果不在新图中绘制，自动启用hold
if ~newfig
    holdon = true;
end

% 创建或使用当前图形
if newfig
    fig = figure;
    set(fig, 'Color', 'white');  % 设置背景为白色
else
    fig = gcf;
    set(fig, 'Color', 'white');  % 确保背景为白色
end
if holdon
    hold on;
end

% 计算频谱
len=fix(length(signal)/2048);
signaltmp=reshape(signal(1:2048*len),2048,len);%将信号重塑为 2048 × len 的矩阵，方便后续批量处理。
fftx=fft(signaltmp);
spectrum=sum(abs(fftx.'));
spectrum=20*log10(spectrum)-max(20*log10(spectrum));
spectrum_shifted = fftshift(spectrum);

% 根据单边/双边谱选择数据
N = 2048;
if onesided_flag
    % 单边谱：只取正频率部分（包括DC）
    % fftshift后，正频率在索引 N/2+1 到 N，对应频率 0 到 baudrate/2
    idx = (N/2+1):N;  % 正频率索引（包括DC，索引从1开始）
    freq = (0:N/2-1)*baudrate/N;  % 频率轴：0 到 baudrate/2
    spectrum_plot = spectrum_shifted(idx);
else
    % 双边谱：全部频率
    freq = (-N/2:N/2-1)*baudrate/N;
    spectrum_plot = spectrum_shifted;
end

% 自动判断频率单位（GHz或Hz）
if max(abs(freq)) >= 1e9
    freq_plot = freq / 1e9;  % 转换为GHz
    freq_unit = 'GHz';
else
    freq_plot = freq;
    freq_unit = 'Hz';
end

% 绘制频谱
if ~isempty(linelabel)
    plot(freq_plot, spectrum_plot, 'Color', linecolor, 'DisplayName', linelabel, 'LineWidth', 2);
else
    plot(freq_plot, spectrum_plot, 'Color', linecolor, 'LineWidth', 2);
end

% 论文级美化：设置字体大小和粗细
set(gca, 'FontSize', 24, 'FontWeight', 'bold', 'LineWidth', 1.5);
xlabel(['Frequency (', freq_unit, ')'], 'FontSize', 24, 'FontWeight', 'bold');
ylabel('Magnitude (dB)', 'FontSize', 24, 'FontWeight', 'bold');

% 设置图例字体
if ~isempty(linelabel) || ~isempty(legend)
    leg = legend('show');
    if ~isempty(leg)
        set(leg, 'FontSize', 24, 'FontWeight', 'bold');
    end
end

% 设置标题（仅在新图时）
% if newfig
%     title('Power Spectral Density', 'FontSize', 20, 'FontWeight', 'bold');
% end

% 美化网格和坐标轴
grid on;
set(gca, 'GridAlpha', 0.2, 'GridLineStyle', '-');  % 设置网格透明度和样式
set(gca, 'Box', 'on');  % 显示坐标轴边框
axis tight;  % 自动调整坐标轴范围
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  200Gb/s Project  Li Zhipei Wang Xishuo Yuan Gao 2021.8 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Plotspectrum(signal,baudrate,spectrum_type,line_color)

% Handle optional parameters
if nargin < 3
    spectrum_type = 2;  % Default: double-sided spectrum
end
if nargin < 4
    line_color = 'b';  % Default: blue
end

number=length(signal);
FN=fftshift(-1/2:1/number:1/2-1/number);
% plot(fftshift(FN)*baudrate,fftshift(10*log10(sqrt(abs(fft(sigx)/length(sigx)).^2))),'b-');

figure
len=fix(length(signal)/2048);
signaltmp=reshape(signal(1:2048*len),2048,len);% Reshape signal to 2048 x len matrix for batch processing
fftx=fft(signaltmp);
spectrum=sum(abs(fftx.'));
spectrum=20*log10(spectrum)-max(20*log10(spectrum));

% Determine frequency axis and spectrum based on spectrum_type
N = 2048;
spectrum_shifted = fftshift(spectrum);
freq_axis = (-N/2:N/2-1)*baudrate/N;

if spectrum_type == 1
    % Single-sided spectrum: only plot right half (positive frequencies)
    % Extract DC and positive frequencies (right half of double-sided spectrum)
    idx_right = N/2+1:N;
    plot(freq_axis(idx_right), spectrum_shifted(idx_right), 'LineWidth', 1.5, 'Color', line_color)
else
    % Double-sided spectrum: full frequency range (-baudrate/2 to baudrate/2)
    plot(freq_axis, spectrum_shifted, 'LineWidth', 1.5, 'Color', line_color)
end

% Set figure background to white
set(gcf, 'Color', 'white')

% Set axis properties, increase font size
ax = gca;
ax.FontSize = 18;  % Increase font size
ax.FontName = 'Times New Roman';  % Use common font for papers
ax.FontWeight = 'bold';  % Bold font for axis tick labels
ax.LineWidth = 1;  % Axis line width
ax.Box = 'on';  % Show axis box

% Add axis labels
xlabel('Frequency (Hz)', 'FontSize', 16, 'FontName', 'Times New Roman', 'FontWeight', 'bold')
ylabel('Normalized Power Spectral Density (dB)', 'FontSize', 16, 'FontName', 'Times New Roman', 'FontWeight', 'bold')

% Set grid for better clarity
grid on
grid minor
ax.GridLineStyle = '-';
ax.GridAlpha = 0.3;
ax.MinorGridLineStyle = ':';
ax.MinorGridAlpha = 0.2;

% Set tick marks for better clarity
ax.TickLength = [0.01, 0.02];
end
function showerrors(title0,Rx_Bit,uhat_16,data_en_bits,data_info_bits)
    % 计算译码前的误码位置
    errors_before_fec = find(Rx_Bit ~= data_en_bits');
    % 计算译码后的误码位置
    errors_after_fec = find(uhat_16 ~= data_info_bits);
    % 绘制2倍采样译码前误码密度图
    figure;
    histogram(errors_before_fec, 'Normalization', 'pdf');
    %% 例如'2倍采样，译码后误码密度分布'

    title(sprintf('%s - 译码前误码密度分布', title0));
    xlabel('比特位置');
    ylabel('密度');
 
    % 绘制译码后误码密度图
    figure;
    histogram(errors_after_fec, 'Normalization', 'pdf');
    title(sprintf('%s - 译码后误码密度分布', title0));
    xlabel('比特位置');
    ylabel('密度');
end


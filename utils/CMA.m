function [Pol_X,epsilon_x]=CMA(x,NoTaps_CMA) 
%% CMA处理
 %% CMA Adaptation
    %NoTaps_CMA= 25;    %   fir滤波器的腔长  13  要奇数
    mu_CMA=0.001;     %0.001  %系数，在0.01和0.001之间变换  %% zx将0.001修改为0.01
    mv_CMA=0.0006;   %0.0006  要比mu_CMA小一些
    ma_CMA=0.0001;    %参考0.0006   
    %% 输入实验数据
    L1=(NoTaps_CMA-1)/2;
    PAM_rx=[x(1:L1);x;x(length(x)-L1+1:length(x))];   %前后各加丢失的6个数据 
    y1=PAM_rx';
    %% ---------------   CMA  -----------------------%%
    P_I=mean(y1.^2);% mean取平均值
    I=y1/sqrt(P_I);  
    Pol_x_Rx=I;     
    %% Stage 2: CMA Polarization Tracking横模偏振跟踪
    CenterIndex=round(NoTaps_CMA/2);      % 四舍五入中心点
    H_xx=[zeros(CenterIndex-1,1); 1; zeros(CenterIndex-1,1)];  %H矩阵 只有第CenterIndex位置为1其余均为0
    MeanPower1=mean(Pol_x_Rx.^2);               %  mean是单行或单列的平均值，若是矩阵，则返回是每列平均值
 %   Pol_x_Rx=Pol_x_Rx./sqrt(MeanPower1/2);      %  输出 平均功率变成 2 
    tmpLength=length(Pol_x_Rx)-NoTaps_CMA+1;    %  由于这边会使数据量丢掉NoTaps_CMA-1个数据,体现在Pol_X_1=H_xx.'*Pol_X_input(:,i);前后各18个数据 
    for ii =1:1:tmpLength  
        Pol_X_input(:,ii)=Pol_x_Rx(ii+NoTaps_CMA-1:-1:ii).'; 
    end
    %% CMA1 %%%%%%%%%%%%%
    mu=mu_CMA; %0.001  %系数，在0.01和0.001之间变换
    for i=1:length(Pol_X_input)
        Pol_X_1=H_xx.'*Pol_X_input(:,i); 
        epsilon_x(i)=1-abs(Pol_X_1)^2;
        H_xx=H_xx+mu*epsilon_x(i)*Pol_X_1*conj(Pol_X_input(:,i)); % 每一步都对 Hxx 进行修正   
    end
        Pol_X=H_xx.'*Pol_X_input; 
    %% %%%%%%%%%%%%%%%%% CMA2 %%%%%%%%%%%%%
    mv=mv_CMA;
    for i=1:length(Pol_X_input)  
        Pol_X(i)=H_xx.'*Pol_X_input(:,i);
        epsilon_x(i)=1-abs(Pol_X(i))^2; %  修改均方误差
        H_xx =H_xx+mv*epsilon_x(i)*Pol_X(i)*conj(Pol_X_input(:,i));  % 继续修正 Hxx
    end
    Pol_X=H_xx .'*Pol_X_input;

     ma=ma_CMA;       %参考0.0006
    for i=1:length(Pol_X_input)
        Pol_X(i)=H_xx.'*Pol_X_input(:,i); 
        epsilon_x(i)=1-abs(Pol_X(i))^2;%mean square error
        H_xx=H_xx+ma*epsilon_x(i)*Pol_X(i)*conj(Pol_X_input(:,i));  % 继续修正 Hxx
    end
    Pol_X=H_xx.'*Pol_X_input;
end


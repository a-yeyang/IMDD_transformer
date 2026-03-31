function [output1] = load1(input1)
    filename = fullfile('E:\chenshiyang\IMDD_Transformer\tx_all_to_vpi.txt');
    SignalAdd = load(filename);
    output1= input1;
    output1.band.E = SignalAdd;
end
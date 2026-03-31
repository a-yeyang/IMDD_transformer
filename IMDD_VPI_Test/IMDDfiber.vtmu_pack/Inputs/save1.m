function save1(Rx, filename)
    receiversignal = Rx.band.E;
    filename = fullfile('E:\chenshiyang\IMDD_Transformer\非线性效应数据\vpi_data_5km_laser0.5.txt');
    save(filename, 'receiversignal', '-ascii');
end
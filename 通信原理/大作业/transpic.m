% 读取图像
img = imread('image.png');
gray_img = rgb2gray(img); % 转换为灰度图像
figure;
imshow(gray_img); title('原始灰度图像');

% 将灰度图像转换为二进制序列
bin_img = de2bi(gray_img(:), 8, 'left-msb')';
bin_img = bin_img(:);

A = 1;                  %载波幅度
fc = 1000;                 %载波频率
Nsam = 10;             %每个码元的采样点数,即采样频率fs
fs  = Nsam;
L_Dseq = length(img);   %码元数目,数值长度
T = 1;                  %码元宽度
dt = T/Nsam;            %波形采样间隔
t = 0:dt:L_Dseq*T-dt;   %波形采样时间点，0开始，以dt为步长
Lt = length(t);         %波形采样点数
ct = A*sin(2*pi*fc*t);  %载波

%--------------------发送端---------------
% 将二进制序列转换为NRZ信号
s_NRZ = rectpulse(double(bin_img), Nsam);
dpSignal = real(dpskmod(bin_img, 2));
dp_NRZ = rectpulse(dpSignal, Nsam);

ask = ammod(s_NRZ,fc,fs);
freqdev = 2;
fsk = fmmod(s_NRZ,fc,fs,freqdev);
phasedev = pi/2;
psk = pmmod(s_NRZ,fc,fs,phasedev);
phasedev = pi/2;
dpsk = pmmod(dp_NRZ,fc,fs,phasedev);


%--------------------信道---------------
%   噪声模拟
% snr = 10;  %信噪比
% ask = addNoise(ask,snr,2);
% fsk = addNoise(fsk,snr,2);
% psk = addNoise(psk,snr,2);
% dpsk = addNoise(dpsk,snr,2);


%--------------------接收端---------------
d_ask = amdemod(ask,fc,fs);
rev_ask = intdump(d_ask,fs);
d_fsk = fmdemod(fsk,fc,fs,freqdev);
rev_fsk = intdump(d_fsk,fs);
d_psk = pmdemod(psk,fc,fs,phasedev);
rev_psk = intdump(d_psk,fs);
d_dpsk = pmdemod(dpsk,fc,fs,phasedev);
rev_dpsk = intdump(d_dpsk,fs);
rev_dpsk = dpskdemod(rev_dpsk,M);

% 还原二进制数据到图像数据, 将解调后的信号转换回二进制序列
% ask
d_ask(d_ask > 0.5) = 1;
d_ask(d_ask <= 0.5) = 0;
d_ask_bin = d_ask(1:Nsam:end);
% fsk
d_fsk(d_fsk > 0.5) = 1;
d_fsk(d_fsk <= 0.5) = 0;
d_fsk_bin = d_fsk(1:Nsam:end);
% psk
d_psk(d_psk > 0.5) = 1;
d_psk(d_psk <= 0.5) = 0;
d_psk_bin = d_psk(1:Nsam:end);
% dpsk
d_dpsk = pmdemod(dpsk_noisy, fc, fs, phasedev);
d_dpsk(d_dpsk > 0.5) = 1;
d_dpsk(d_dpsk <= 0.5) = 0;
d_dpsk_bin = d_dpsk(1:Nsam:end);

% 将二进制序列转换回图像
d_ask_img = reshape(bi2de(reshape(d_ask_bin, 8, []).', 'left-msb'), size(gray_img));
d_fsk_img = reshape(bi2de(reshape(d_fsk_bin, 8, []).', 'left-msb'), size(gray_img));
d_psk_img = reshape(bi2de(reshape(d_psk_bin, 8, []).', 'left-msb'), size(gray_img));
d_dpsk_img = reshape(bi2de(reshape(d_dpsk_bin, 8, []).', 'left-msb'), size(gray_img));



figure();
subplot(511);
imshow(image);title('原图像'); 
subplot(512);
imshow(uint8(d_ask_img)); title('ASK解调图像');
subplot(513);
imshow(uint8(d_fsk_img)); title('FSK解调图像');
subplot(514);
imshow(uint8(d_psk_img)); title('PSK解调图像');
subplot(515);
imshow(uint8(d_dpsk_img)); title('DPSK解调图像');


function noisy_signal = addNoise(signal,snr,sel)
    % 添加高斯白噪声
    if sel==1
    noisy_signal = awgn(signal,snr);
    end
    noisy_signal = signal;
    salt = max(signal);
    pepper = min(signal);
    len = uint8(length(signal)/(snr));
    index = randi(length(signal)-1, len);
    for i=1:len
        sVal = randi([0,1]);
        if sVal==0
            noisy_signal(index(i)) = pepper;
        else
            noisy_signal(index(i)) = salt;
        end
    end
end

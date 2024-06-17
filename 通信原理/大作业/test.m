
image = imread('image.png');
image = imresize(image,[256,256]);
grayImage = rgb2gray(image);
binaryImage = imbinarize(grayImage);
% binaryImage = grayImage;
img = double(reshape(binaryImage,1,[]));
% dimg = reshape(img,256,256);
%imshow(double(binaryImage)*255);

M = max(img)+1;  %量级
A = 1;                  %载波幅度
fc = 3;                 %载波频率
Nsam = 120;             %每个码元的采样点数,即采样频率fs
fs  =Nsam;
L_Dseq = length(img);             %码元数目,数值长度
T = 1;                  %码元宽度
dt = T/Nsam;            %波形采样间隔
t = 0:dt:L_Dseq*T-dt;   %波形采样时间点，0开始，以dt为步长
Lt = length(t);         %波形采样点数
ct = A*sin(2*pi*fc*t);  %载波

%--------------------发送端---------------
dpSignal = real(dpskmod(img,M));
s_NRZ = rectpulse(img,Nsam);
d_NRZ = s_NRZ*2-1;
dp_NRZ = rectpulse(dpSignal,Nsam);

ask = ammod(s_NRZ,fc,fs);
freqdev = 2;
fsk = fmmod(s_NRZ,fc,fs,freqdev);
phasedev = pi/2;
psk = pmmod(s_NRZ,fc,fs,phasedev);
phasedev = pi/2;
dpsk = pmmod(dp_NRZ,fc,fs,phasedev);


%--------------------信道---------------
%   1是高斯白噪声 2是椒盐噪声（存在问题）
snr = 10;  %信噪比
ask = addNoise(ask,snr,2);
fsk = addNoise(fsk,snr,2);
psk = addNoise(psk,snr,2);
dpsk = addNoise(dpsk,snr,2);


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

imAsk =  reshape(rev_ask,256,256);
imFsk =  reshape(rev_fsk,256,256);
imPsk =  reshape(rev_psk,256,256);
imDPsk =  reshape(rev_dpsk,256,256);

figure();
subplot(321);
imshow(image);title('原图像'); 
subplot(322);
imshow(double(binaryImage)*255);title('位图'); 
subplot(323);
imshow(imAsk);title('ASK图像'); 
subplot(324);
imshow(imFsk);title('FSK图像'); 
subplot(325);
imshow(imPsk);title('PSK图像'); 
subplot(326);
imshow(imDPsk);title('DPSK图像'); 


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





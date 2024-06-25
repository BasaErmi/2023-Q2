% 读取图像
img = imread('image.png'); % 替换为你的图像文件路径
gray_img = rgb2gray(img); % 转换为灰度图像


% 将灰度图像转换为二进制序列
bin_img = de2bi(gray_img(:), 8, 'left-msb')';
bin_img = bin_img(:);

% 参数设置
A = 1;                  % 载波幅度
fc = 30;              % 载波频率
Nsam = 4;              % 每个码元的采样点数,即采样频率fs
fs = Nsam * fc;
L_Dseq = length(bin_img); % 码元数目,数值长度
T = 1;                  % 码元宽度
dt = T/Nsam;            % 波形采样间隔
t = 0:dt:(L_Dseq*T-dt); % 波形采样时间点，0开始，以dt为步长
Lt = length(t);         % 波形采样点数

% 将二进制序列转换为NRZ信号
s_NRZ = rectpulse(double(bin_img), Nsam);

% 调制
ask = ammod(s_NRZ, fc, fs);

% 添加噪声
SNR = 100; % 信噪比，单位为dB
lambda = 0.01;
noise_density = 0.1
ask_noisy = add_salt_and_pepper_noise(ask, noise_density);

% 解调
d_ask = amdemod(ask_noisy, fc, fs);

% 将解调后的信号转换回二进制序列
d_ask(d_ask > 0.5) = 1;
d_ask(d_ask <= 0.5) = 0;
d_ask_bin = d_ask(1:Nsam:end);

% 将二进制序列转换回图像
d_img = reshape(bi2de(reshape(d_ask_bin, 8, []).', 'left-msb'), size(gray_img));

% 显示解调后的图像
figure;
subplot(151);
imshow(gray_img); title('原图');
subplot(152);
imshow(uint8(d_img)); title('ASK解调图像');

% 频移键控调制和解调
freqdev = 2;
fsk = fmmod(s_NRZ, fc, fs, freqdev);
fsk_noisy = add_salt_and_pepper_noise(fsk, noise_density);
d_fsk = fmdemod(fsk_noisy, fc, fs, freqdev);
d_fsk(d_fsk > 0.5) = 1;
d_fsk(d_fsk <= 0.5) = 0;
d_fsk_bin = d_fsk(1:Nsam:end);
d_fsk_img = reshape(bi2de(reshape(d_fsk_bin, 8, []).', 'left-msb'), size(gray_img));

subplot(153);
imshow(uint8(d_fsk_img)); title('FSK解调图像');

% 相位调制和解调
phasedev = pi/2;
psk = pmmod(s_NRZ, fc, fs, phasedev);
psk_noisy = add_salt_and_pepper_noise(psk, noise_density);
d_psk = pmdemod(psk_noisy, fc, fs, phasedev);
d_psk(d_psk > 0.5) = 1;
d_psk(d_psk <= 0.5) = 0;
d_psk_bin = d_psk(1:Nsam:end);
d_psk_img = reshape(bi2de(reshape(d_psk_bin, 8, []).', 'left-msb'), size(gray_img));

subplot(154);
imshow(uint8(d_psk_img)); title('PSK解调图像');

% 差分相移键控 (DPSK) 调制和解调
dpsk = dpskmod(bin_img, 2); % DPSK调制
dpsk_noisy = add_salt_and_pepper_noise(dpsk, noise_density);
d_dpsk = dpskdemod(dpsk_noisy, 2); % DPSK解调

% 将解调后的二进制序列转换回图像
d_dpsk_img = reshape(bi2de(reshape(d_dpsk, 8, []).', 'left-msb'), size(gray_img));

subplot(155);
imshow(uint8(d_dpsk_img)); title('DPSK解调图像');

% 计算均方误差（MSE）
mse_value_ask = immse(uint8(d_img), gray_img);
% 计算峰值信噪比（PSNR）
psnr_value_ask = psnr(uint8(d_img), gray_img);

% 对FSK调制解调结果计算误差
mse_value_fsk = immse(uint8(d_fsk_img), gray_img);
psnr_value_fsk = psnr(uint8(d_fsk_img), gray_img);

% 对PSK调制解调结果计算误差
mse_value_psk = immse(uint8(d_psk_img), gray_img);
psnr_value_psk = psnr(uint8(d_psk_img), gray_img);

% 对DPSK调制解调结果计算误差
mse_value_dpsk = immse(uint8(d_dpsk_img), gray_img);
psnr_value_dpsk = psnr(uint8(d_dpsk_img), gray_img);

% 显示所有解调图像和误差值
figure;

subplot(221);
imshow(uint8(d_img)); title(['ASK解调图像, MSE: ' num2str(mse_value_ask) ', PSNR: ' num2str(psnr_value_ask) ' dB']);
subplot(222);
imshow(uint8(d_fsk_img)); title(['FSK解调图像, MSE: ' num2str(mse_value_fsk) ', PSNR: ' num2str(psnr_value_fsk) ' dB']);
subplot(223);
imshow(uint8(d_psk_img)); title(['PSK解调图像, MSE: ' num2str(mse_value_psk) ', PSNR: ' num2str(psnr_value_psk) ' dB']);
subplot(224);
imshow(uint8(d_dpsk_img)); title(['DPSK解调图像, MSE: ' num2str(mse_value_dpsk) ', PSNR: ' num2str(psnr_value_dpsk) ' dB']);


% 添加噪声函数

% 添加瑞丽噪声
function noisy_signal = add_rayleigh_noise(signal, snr)
    % Generate Rayleigh noise
    noise = raylrnd(1, size(signal));
    % Normalize noise to match desired SNR
    signal_power = rms(signal)^2;
    noise_power = rms(noise)^2;
    scaling_factor = sqrt(signal_power / (10^(snr / 10) * noise_power));
    noisy_signal = signal + noise * scaling_factor;
end

% 添加椒盐噪声
function noisy_signal = add_salt_and_pepper_noise(signal, noise_density)
    noisy_signal = signal;
    num_samples = numel(signal);
    num_noisy_samples = round(noise_density * num_samples);
    
    % 生成随机索引
    indices = randperm(num_samples, num_noisy_samples);
    
    % 随机设置一些样本为最大值或最小值
    for i = 1:num_noisy_samples
        if rand < 0.5
            noisy_signal(indices(i)) = max(signal); % Salt noise
        else
            noisy_signal(indices(i)) = min(signal); % Pepper noise
        end
    end
end

% 添加泊松噪声
function noisy_signal = add_poisson_noise(signal, lambda)
    % 生成泊松噪声
    noise = poissrnd(lambda, size(signal)) - lambda;
    % 将噪声添加到信号中
    noisy_signal = signal + noise;
end
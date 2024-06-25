
A = 1;                  %载波幅度
fc = 24;                 %载波频率
Nsam = 120;              %每个码元的采样点数,即采样频率fs
fs  =Nsam;
L_Dseq = 7;             %码元数目,数值长度
T = 1;                  %码元宽度
dt = T/Nsam;            %波形采样间隔
t = 0:dt:L_Dseq*T-dt;   %波形采样时间点，0开始，以dt为步长
Lt = length(t);         %波形采样点数
ct = A*sin(2*pi*fc*t);  %载波

binSignal = [0,1,0,0,1,1,0];
dpSignal = real(dpskmod(binSignal,2));
s_NRZ = rectpulse(binSignal,Nsam);
d_NRZ = s_NRZ*2-1;
dp_NRZ = rectpulse(dpSignal,Nsam);


% 添加噪声
SNR = 10; % 信噪比，单位为dB
noise_density = 0.03; % 噪声密度

lambda = 0.05;  % 泊松噪声强度
poissonNoise = poissrnd(lambda, size(dp_NRZ)); % 生成泊松噪声

ask = ammod(s_NRZ,fc,fs);
ask_noisy = awgn(ask, SNR, "measured");
d_ask = amdemod(ask_noisy,fc,fs);

freqdev = 2;
fsk = fmmod(s_NRZ,fc,fs,freqdev);
fsk_noisy = awgn(fsk, SNR, "measured");
d_fsk = fmdemod(fsk_noisy,fc,fs,freqdev);

phasedev = pi/2;
psk = pmmod(s_NRZ,fc,fs,phasedev);
psk_noisy = awgn(psk, SNR, "measured");
d_psk = pmdemod(psk_noisy,fc,fs,phasedev);
phasedev = pi/2;
dpsk = pmmod(dp_NRZ,fc,fs,phasedev);
dpsk_noisy = awgn(dpsk, SNR, "measured");
d_dpsk = pmdemod(dpsk_noisy,fc,fs,phasedev);
rev_dpsk = intdump(d_dpsk,fs); 
rev_dpsk = dpskdemod(rev_dpsk,2); 
rev_dpsk = rectpulse(rev_dpsk,Nsam);


figure(); 
subplot(411);
plot(t,d_ask,'LineWidth',2);title("ASK解调波形");
subplot(412);
plot(t,d_fsk,'LineWidth',2);title("FSK解调波形");
subplot(413);
plot(t,d_psk,'LineWidth',2);title("PSK解调波形");
subplot(414);
plot(t,rev_dpsk,'LineWidth',2);title("DPSK解调波形");


% 添加瑞丽噪声函数
function noisy_signal = add_rayleigh_noise(signal, snr)
    % Generate Rayleigh noise
    noise = raylrnd(1, size(signal));
    % Normalize noise to match desired SNR
    signal_power = rms(signal)^2;
    noise_power = rms(noise)^2;
    scaling_factor = sqrt(signal_power / (10^(snr / 10) * noise_power));
    noisy_signal = signal + noise * scaling_factor;
end

% 添加椒盐噪声函数
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

% 添加泊松噪声函数
function noisy_signal = add_poisson_noise(signal, lambda)
    % 生成泊松噪声
    noise = poissrnd(lambda, size(signal)) - lambda;
    % 将噪声添加到信号中
    noisy_signal = signal + noise;
end
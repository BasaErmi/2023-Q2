
A = 1;                  %载波幅度
fc = 24;                 %载波频率
Nsam = 60;            %每个码元的采样点数,即采样频率fs
fs  =Nsam;
L_Dseq = 7;             %码元数目,数值长度
T = 1;                  %码元宽度
dt = T/Nsam;            %波形采样间隔
t = 0:dt:L_Dseq*T-dt;   %波形采样时间点，0开始，以dt为步长
Lt = length(t);         %波形采样点数
ct = A*sin(2*pi*fc*t);  %载波


 
%信源和矩阵成波
binSignal = [0,1,0,0,1,1,0];
% dpinit = 0;
% dpSignal = zeros(L_Dseq);
% pre = dpinit;
% for i = 1:L_Dseq
%     cur = xor(pre,binSignal(i));
%     dpSignal(i) = cur;
%     pre = cur;
% end
dpSignal = real(dpskmod(binSignal,2));




%单极性
s_NRZ = rectpulse(binSignal,Nsam);
%双极性
d_NRZ = s_NRZ*2-1;
%差分码
dp_NRZ = rectpulse(dpSignal,Nsam);

figure(); 
subplot(311);
plot(t,s_NRZ,'LineWidth',2);title('单极性波形'); 
subplot(312);
plot(t,d_NRZ,'LineWidth',2);title("双极性波形");
subplot(313);
plot(t,dp_NRZ,'LineWidth',2);title("差分波形");



%----------------2ask---------------
ask = ammod(s_NRZ,fc,fs);
d_ask = amdemod(ask,fc,fs);
figure(); 
subplot(311);
plot(t,s_NRZ,'LineWidth',2);title('原波形'); 
subplot(312);
plot(t,ask,'LineWidth',2);title("ASK调制波形");
subplot(313);
plot(t,d_ask,'LineWidth',2);title("ASK解调波形");



%----------------2fsk---------------
%s_2fsk = A*sin(2*pi*(fc+d_NRZ).*t);
%plot(t,s_2fsk,'LineWidth',2);
freqdev = 2;
fsk = fmmod(s_NRZ,fc,fs,freqdev);
d_fsk = fmdemod(fsk,fc,fs,freqdev);
figure(); 
subplot(311);
plot(t,s_NRZ,'LineWidth',2);title('原波形'); 
subplot(312);
plot(t,fsk,'LineWidth',2);title("FSK调制波形");
subplot(313);
plot(t,d_fsk,'LineWidth',2);title("FSK解调波形");
res = intdump(d_fsk,fs);

%----------------2psk---------------
phasedev = pi/2;
psk = pmmod(s_NRZ,fc,fs,phasedev);
d_psk = pmdemod(psk,fc,fs,phasedev);
figure(); 
subplot(311);
plot(t,s_NRZ,'LineWidth',2);title('原波形');
subplot(312);
plot(t,psk,'LineWidth',2);title("PSK调制波形");
subplot(313);
plot(t,d_psk,'LineWidth',2);title("PSK解调波形");

%----------------2dpsk---------------

phasedev = pi/2;
dpsk = pmmod(dp_NRZ,fc,fs,phasedev);
d_dpsk = pmdemod(dpsk,fc,fs,phasedev);
rev_dpsk = intdump(d_dpsk,fs); 
rev_dpsk = dpskdemod(rev_dpsk,2); 
rev_dpsk = rectpulse(rev_dpsk,Nsam);

figure(); 
subplot(311);
plot(t,s_NRZ,'LineWidth',2);title('原波形');
subplot(312);
plot(t,dpsk,'LineWidth',2);title("DPSK调制波形");
subplot(313);
plot(t,rev_dpsk,'LineWidth',2);title("DPSK解调波形");


ddpsk = dpskdemod(d_dpsk,2);
figure(3); 
subplot(311);
plot(t,s_NRZ,'LineWidth',2);
phasedev = pi/2;
Dpsk = dpskmod(binSignal)
subplot(312);
plot(t,psk,'LineWidth',2);
dpsk = pmdemod(psk,fc,fs,phasedev);
subplot(313);
plot(t,dpsk,'LineWidth',2);

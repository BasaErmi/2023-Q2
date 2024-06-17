
A = 1;                  %载波幅度
fc = 3;                 %载波频率
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

ask = ammod(s_NRZ,fc,fs);
d_ask = amdemod(ask,fc,fs);
freqdev = 2;
fsk = fmmod(s_NRZ,fc,fs,freqdev);
d_fsk = fmdemod(fsk,fc,fs,freqdev);
phasedev = pi/2;
psk = pmmod(s_NRZ,fc,fs,phasedev);
d_psk = pmdemod(psk,fc,fs,phasedev);
phasedev = pi/2;
dpsk = pmmod(dp_NRZ,fc,fs,phasedev);
d_dpsk = pmdemod(dpsk,fc,fs,phasedev);



figure(); 
subplot(521);
plot(t,s_NRZ,'LineWidth',2);title('极性波形'); 
subplot(522);
plot(t,dp_NRZ,'LineWidth',2);title("差分波形");
subplot(523);
plot(t,fsk,'LineWidth',2);title("FSK调制波形");
subplot(525);
plot(t,ask,'LineWidth',2);title("ASK调制波形");
subplot(527);
plot(t,psk,'LineWidth',2);title("PSK调制波形");
subplot(529);
plot(t,dpsk,'LineWidth',2);title("DPSK调制波形");
subplot(524);
plot(t,d_ask,'LineWidth',2);title("ASK解调波形");
subplot(526);
plot(t,d_fsk,'LineWidth',2);title("FSK解调波形");
subplot(528);
plot(t,d_psk,'LineWidth',2);title("PSK解调波形");
subplot(5,2,10);
plot(t,d_dpsk,'LineWidth',2);title("DPSK解调波形");







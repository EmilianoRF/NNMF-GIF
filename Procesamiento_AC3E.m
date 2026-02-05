datos = load('Repositorio AC3E\Controles\FN001\R02 - Calibrated Gestures\vowels.mat');
fsr = 8000;
fs  = 20000;

ACC = datos.ACC;
MBK = datos.MBK;
EGG = datos.EGG;
OVV = datos.OVV;

tiempo = 0:1/fs:(length(ACC.data)-1)/fs;
figure
subplot(411)
plot(tiempo,ACC.data)
legend('ACC')
subplot(412)
plot(tiempo,MBK.data)
legend('MBK')
subplot(413)
plot(tiempo,EGG.data)
legend('EGG')
subplot(414)
plot(tiempo,OVV.data)
legend('OVV')



audiowrite('mic.wav',MBK.data,8000)
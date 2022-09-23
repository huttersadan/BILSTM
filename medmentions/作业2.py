from scipy.signal import lti,step2,impulse2
import matplotlib.pyplot as plt

Kpa = -100
Kda = 10
Kdx = 20
Kpx = 8

fenzi = [-0.25*Kda,-(25+0.25*Kpa)]
fenmu = [1,0.25*Kdx,0.25*Kpx]

s1=lti([fenzi[0],fenzi[1]],[fenmu[0],fenmu[1],fenmu[2]]) # 以分子分母的最高次幂降序的系数构建传递函数，s1=3/(s^2+2s+10）

t1,y1=step2(s1)
# 计算阶跃输出，y1是Step response of system.

f,((ax1)) = plt.subplots(1,1,sharex='col',sharey='row') # 开启subplots模式
ax1.plot(t1,y1,'r',label='s1 Step Response',linewidth=0.5)

##plt.xlabel('Times')
##plt.ylabel('Amplitude')
#plt.legend()
plt.show()
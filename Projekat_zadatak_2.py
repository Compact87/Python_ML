import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.pyplot import figure
from scipy import linalg
pd.options.display.float_format = '{:.2f}'.format

imena = ['pca1','pca2', 'klasa']
data=pd.read_csv('iris_redukovano.csv',names=imena)



s1=data.iloc[:40,0].values
s2=data.iloc[:40,1].values
s3=data.iloc[:40,2].values

y1=data.iloc[50:90,0].values
y2=data.iloc[50:90,1].values
y3=data.iloc[50:90,2].values

z1=data.iloc[100:140,0].values
z2=data.iloc[100:140,1].values
z3=data.iloc[100:140,2].values



M1=np.array([s1.mean(),s2.mean()])
M2=np.array([y1.mean(),y2.mean()])
M3=np.array([z1.mean(),z2.mean()])


m11=M1[0]
m12=M1[1]
m21=M2[0]
m22=M2[1]
m31=M3[0]
m32=M3[1]


N=40
setosa=data[data['klasa']==0]
versicolor=data[data['klasa']==1]
virginica=data[data['klasa']==2]

Sigmax=setosa.iloc[:,0:2].cov().to_numpy()
Sigmay=versicolor.iloc[:,0:2].cov().to_numpy()
Sigmaz=virginica.iloc[:,0:2].cov().to_numpy()


invSigmax=linalg.inv(Sigmax)
invSigmay=linalg.inv(Sigmay)
invSigmaz=linalg.inv(Sigmaz)

x11=invSigmax[0][0]
x12=invSigmax[0][1]
x21=invSigmax[1][0]
x22=invSigmax[1][1]

y11=invSigmay[0][0]
y12=invSigmay[0][1]
y21=invSigmay[1][0]
y22=invSigmay[1][1]

z11=invSigmaz[0][0]
z12=invSigmaz[0][1]
z21=invSigmaz[1][0]
z22=invSigmaz[1][1]

X=np.arange(-4,4,0.1)
Y=np.arange(-2,2,0.1)
x1,x2=np.meshgrid(X,Y)

xy=0.5*(((x1-m11)*x11+(x2-m12)*x21)*(x1-m11)+((x1-m11)*x12+(x2-m12)*x22)*(x2-m12))-0.5*(((x1-m21)*y11+(x2-m22)*y21)*(x1-m21)+((x1-m21)*y12+(x2-m22)*y22)*(x2-m22))+0.5*np.log(np.linalg.det(Sigmax)/np.linalg.det(Sigmay))
yz=0.5*(((x1-m21)*y11+(x2-m22)*y21)*(x1-m21)+((x1-m21)*y12+(x2-m22)*y22)*(x2-m22))-0.5*(((x1-m31)*z11+(x2-m32)*z21)*(x1-m31)+((x1-m31)*z12+(x2-m32)*z22)*(x2-m32))+0.5*np.log(np.linalg.det(Sigmay)/np.linalg.det(Sigmaz))
xz=0.5*(((x1-m11)*x11+(x2-m12)*x21)*(x1-m11)+((x1-m11)*x12+(x2-m12)*x22)*(x2-m12))-0.5*(((x1-m31)*z11+(x2-m32)*z21)*(x1-m31)+((x1-m31)*z12+(x2-m32)*z22)*(x2-m32))+0.5*np.log(np.linalg.det(Sigmax)/np.linalg.det(Sigmaz))


fig = plt.figure()
plt.plot(y1, y2, '*m',label='Versicilor')
plt.plot(z1, z2, '.c',label='Virginica')
plt.plot(s1, s2, 'ob',label='Setosa')
plt.contour(x1,x2,yz,0,colors='red')
plt.contour(x1,x2,xy,0,colors='blue')
plt.contour(x1,x2,xz,0)
plt.grid(True)
plt.title('Diskriminacione linije')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
fig.savefig('zadatak2-1-0.png')


set1=data.iloc[40:49,0].values
set2=data.iloc[40:49,1].values
set3=data.iloc[40:49,2].values

ver1=data.iloc[90:99,0].values
ver2=data.iloc[90:99,1].values
ver3=data.iloc[90:99,2].values

vir1=data.iloc[140:149,0].values
vir2=data.iloc[140:149,1].values
vir3=data.iloc[140:149,2].values


#################################### Procena greske za Setosu i Versicolor#########################

x1p1=set1
x2p1=set2

h1=0.5*(((x1p1-m11)*x11+(x2p1-m12)*x21)*(x1p1-m11)+((x1p1-m11)*x12+(x2p1-m12)*x22)*(x2p1-m12))-0.5*(((x1p1-m21)*y11+(x2p1-m22)*y21)*(x1p1-m21)+((x1p1-m21)*y12+(x2p1-m22)*y22)*(x2p1-m22))+0.5*np.log(np.linalg.det(Sigmax)/np.linalg.det(Sigmay))

greska1=0
for i in h1:
   if i>0: #tamo gde nam je diskriminaciona prava za odbirke iz prve␣→klase veća od nule do znači da nam odbirci nisu dobro klasifikovani, jer za␣,→prvu klasu mora biti manja od nule
      greska1=greska1+1


x1p1=ver1
x2p1=ver2

h2=0.5*(((x1p1-m11)*x11+(x2p1-m12)*x21)*(x1p1-m11)+((x1p1-m11)*x12+(x2p1-m12)*x22)*(x2p1-m12))-0.5*(((x1p1-m21)*y11+(x2p1-m22)*y21)*(x1p1-m21)+((x1p1-m21)*y12+(x2p1-m22)*y22)*(x2p1-m22))+0.5*np.log(np.linalg.det(Sigmax)/np.linalg.det(Sigmay))

greska2=0
for i in h2:
   if i<0: #tamo gde nam je diskriminaciona prava za odbirke iz prve␣→klase veća od nule do znači da nam odbirci nisu dobro klasifikovani, jer za␣,→prvu klasu mora biti manja od nule
      greska2=greska2+1


Eps1=((greska1)/9)
Eps2=((greska2)/9)
P=0.5*Eps1+0.5*Eps2 #Ukupna Bajesova greska
T=(1-P)*100 #Tacnost klasifikatora
print(T)
print(greska1,greska2)
fig = plt.figure()
plt.plot(set1, set2, 'ob',label='Setosa')
plt.plot(ver1, ver2, '*m',label='Versicolor')
plt.contour(x1,x2,xy,0)
plt.grid(True)
plt.title('Setosa i versicolor klasifikator test')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
fig.savefig('zadatak2SetosaVers.png')
##################################### Procena greske za Versicolor i Virginiku#############

x1p1=ver1
x1p2=ver2

h1=0.5*(((x1p1-m21)*y11+(x2p1-m22)*y21)*(x1p1-m21)+((x1p1-m21)*y12+(x2p1-m22)*y22)*(x2p1-m22))-0.5*(((x1p1-m31)*z11+(x2p1-m32)*z21)*(x1p1-m31)+((x1p1-m31)*z12+(x2p1-m32)*z22)*(x2p1-m32))+0.5*np.log(np.linalg.det(Sigmay)/np.linalg.det(Sigmaz))

greska3=0
for i in h1:
   if i>0: 
      greska3=greska3+1


x1p1=vir1
x2p1=vir2

h2=0.5*(((x1p1-m21)*y11+(x2p1-m22)*y21)*(x1p1-m21)+((x1p1-m21)*y12+(x2p1-m22)*y22)*(x2p1-m22))-0.5*(((x1p1-m31)*z11+(x2p1-m32)*z21)*(x1p1-m31)+((x1p1-m31)*z12+(x2p1-m32)*z22)*(x2p1-m32))+0.5*np.log(np.linalg.det(Sigmay)/np.linalg.det(Sigmaz))

greska4=0
for i in h2:
   if i<0:
      greska4=greska4+1
fig = plt.figure()
plt.plot(ver1, ver2, '*m',label='Versicolor')
plt.plot(vir1,vir2,'.c',label='Virginica')
plt.contour(x1,x2,yz,0)
plt.grid(True)
plt.title('Versicolor i Viriginica klasifikator test')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
fig.savefig('zadatak2-VersVirg.png')      

Eps1=((greska3)/9)
Eps2=((greska4)/9)
P=0.5*Eps1+0.5*Eps2 #Ukupna Bajesova greska
T=(1-P)*100 #Tacnost klasifikatora
print(T)
print(greska3,greska4)
######################################################setosa VIRG##############

x1p1=set1
x1p2=set2

h1=0.5*(((x1p1-m21)*y11+(x2p1-m22)*y21)*(x1p1-m21)+((x1p1-m21)*y12+(x2p1-m22)*y22)*(x2p1-m22))-0.5*(((x1p1-m31)*z11+(x2p1-m32)*z21)*(x1p1-m31)+((x1p1-m31)*z12+(x2p1-m32)*z22)*(x2p1-m32))+0.5*np.log(np.linalg.det(Sigmay)/np.linalg.det(Sigmaz))

greska3=0
for i in h1:
   if i>0: 
      greska5=greska3+1


x1p1=vir1
x2p1=vir2

h2=0.5*(((x1p1-m21)*y11+(x2p1-m22)*y21)*(x1p1-m21)+((x1p1-m21)*y12+(x2p1-m22)*y22)*(x2p1-m22))-0.5*(((x1p1-m31)*z11+(x2p1-m32)*z21)*(x1p1-m31)+((x1p1-m31)*z12+(x2p1-m32)*z22)*(x2p1-m32))+0.5*np.log(np.linalg.det(Sigmay)/np.linalg.det(Sigmaz))

greska6=0
for i in h2:
   if i<0:  
      greska4=greska4+1
fig = plt.figure()
plt.plot(set1, set2, '*m',label='Setosa')
plt.plot(vir1,vir2,'.c',label='Virginica')
plt.contour(x1,x2,xz,0)
plt.grid(True)
plt.title('Setosa i Viriginica klasifikator test')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
fig.savefig('zadatak2-SetVirg.png')      
##################################################




fig = plt.figure()
plt.plot(set1, set2, 'ob',label='Setosa')
plt.plot(ver1, ver2, '*m',label='Versicolor')
plt.plot(vir1,vir2,'.c',label='Virginica')
plt.contour(x1,x2,xy,0)
plt.contour(x1,x2,yz,0)
plt.grid(True)
plt.title('Prikaz dve klase u x1 x2 prostoru')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
fig.savefig('zadatak2-1.png')



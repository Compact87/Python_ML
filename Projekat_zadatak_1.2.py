import numpy as np
import matplotlib.pyplot as plt

M1=np.array([1,1])#vektor srednjih vrednosti
M2=np.array([5,2])#vektror srednjih vrrednosti
Sigma1=np.array([[2,1],[1,1]])#kovariaciona matrica, prve klase
Sigma2=np.array([[3,-1],[-1,2]])#kovariaciona matrica druge klase
np.random.seed(0)
N=700
P1=P2=0.5
y1, y2=np.random.multivariate_normal(M1,Sigma1,N).T
z1, z2=np.random.multivariate_normal(M2,Sigma2,N).T


from scipy import linalg
invSigma1=linalg.inv(Sigma1)
invSigma2=linalg.inv(Sigma2)
M3=M2.T-M1.T
m1=M3[0]
m2=M3[1]
m11=M1[0]
m12=M1[1]
m21=M2[0]
m22=M2[1]
z11=invSigma1[0][0]
z12=invSigma1[0][1]
z21=invSigma1[1][0]
z22=invSigma1[1][1]

k11=invSigma2[0][0]
k12=invSigma2[0][1]
k21=invSigma2[1][0]
k22=invSigma2[1][1]



######################## D2  KRIVE KLASA 1 ###################################
s1 = np.linspace(-5, 11, 120) #u odnosu na x-osu 2D slučajnog vektora formiramo raspon tačaka u proizvoljnom brojutačaka, ovde je uzeto 120
s2 = np.linspace(-5, 11, 120)
x1pom,x2pom=np.meshgrid(s1,s2)
d1=(invSigma1[0,0]*(x1pom-M1[0])+invSigma1[0,1]*(x2pom-M1[1]))*(x1pom-M1[0])+(invSigma1[1,0]*(x1pom-M1[0])+invSigma1[1,1]*(x2pom-M1[1]))*(x2pom-M1[1]) 
d2=(invSigma2[0,0]*(x1pom-M2[0])+invSigma2[0,1]*(x2pom-M2[1]))*(x1pom-M2[0])+(invSigma2[1,0]*(x1pom-M2[0])+invSigma2[1,1]*(x2pom-M2[1]))*(x2pom-M2[1]) #formula uzeta iz predavanja
fig = plt.figure()
plt.plot(y1,y2,'m*')
plt.plot(z1,z2,'c.')
plt.contour(x1pom,x2pom,d2,[1,4,9],colors='k',linewidths=3)
plt.contour(x1pom,x2pom,d1,[1,4,9],colors='k',linewidths=3)
plt.title('Dvodimenzioni slucajni vektor i d^2 krive')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.show()
fig.savefig('zadatak1-d2linije.png')

############################## PRIKAZ KLASA I DISKRIMINACIONE LINIJE #############################################
X=np.arange(-2,7,0.1)
Y=np.arange(-2,7,0.1)
x1,x2=np.meshgrid(X,Y)

y=0.5*(((x1-m11)*z11+(x2-m12)*z21)*(x1-m11)+((x1-m11)*z12+(x2-m12)*z22)*(x2-m12))-0.5*(((x1-m21)*k11+(x2-m22)*k21)*(x1-m21)+((x1-m21)*k12+(x2-m22)*k22)*(x2-m22))+0.5*np.log(np.linalg.det(Sigma1)/np.linalg.det(Sigma2))+np.log(P1/P2)
l=0.5*(((x1-m11)*z11+(x2-m12)*z21)*(x1-m11)+((x1-m11)*z12+(x2-m12)*z22)*(x2-m12))-0.5*(((x1-m21)*k11+(x2-m22)*k21)*(x1-m21)+((x1-m21)*k12+(x2-m22)*k22)*(x2-m22))+0.5*np.log(np.linalg.det(Sigma1)/np.linalg.det(Sigma2))+np.log(0.2/0.8)
b=0.5*(((x1-m11)*z11+(x2-m12)*z21)*(x1-m11)+((x1-m11)*z12+(x2-m12)*z22)*(x2-m12))-0.5*(((x1-m21)*k11+(x2-m22)*k21)*(x1-m21)+((x1-m21)*k12+(x2-m22)*k22)*(x2-m22))+0.5*np.log(np.linalg.det(Sigma1)/np.linalg.det(Sigma2))+np.log(0.8/0.2)
fig = plt.figure()

plt.plot(y1, y2, '*m',label='Klasa I')
plt.plot(z1, z2, '.c',label='Klasa II')
plt.contour(x1,x2,y,0)
plt.contour(x1,x2,l,0,colors='red')
plt.contour(x1,x2,b,0,colors='blue')

plt.grid(True)
plt.title('Prikaz dve klase u x1 x2 prostoru')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
fig.savefig('zadatak1-discKrive.png')


####################################### TESTIRANJE NOVIM KLASAMA #####################
M11 =np.array([1.1,1.01])
M21 =np.array([5.04, 2.2])
SigmaM1 =np.array([[2,1],[1,1]])
SigmaM2 =np.array([[3,-1],[-1,2]])
N1=60

y111, y211 = np.random.multivariate_normal(M11, SigmaM1, N1).T
z111, z211 = np.random.multivariate_normal(M21, SigmaM2, N1).T




x1p1=y111
x2p1=y211

h1=0.5*(((x1p1-m11)*z11+(x2p1-m12)*z21)*(x1p1-m11)+((x1p1-m11)*z12+(x2p1-m12)*z22)*(x2p1-m12))-0.5*(((x1p1-m21)*k11+(x2p1-m22)*k21)*(x1p1-m21)+((x1p1-m21)*k12+(x2p1-m22)*k22)*(x2p1-m22))+0.5*np.log(np.linalg.det(Sigma1)/np.linalg.det(Sigma2)) + np.log(P1/P2)

greska1=0
for i in h1:
   if i>0: #tamo gde nam je diskriminaciona prava za odbirke iz prve␣→klase veća od nule do znači da nam odbirci nisu dobro klasifikovani, jer za␣,→prvu klasu mora biti manja od nule
      greska1=greska1+1



x1p2=z111
x2p2=z211

h2=0.5*(((x1p2-m11)*z11+(x2p2-m12)*z21)*(x1p2-m11)+((x1p2-m11)*z12+(x2p2-m12)*z22)*(x2p2-m12))-0.5*(((x1p2-m21)*k11+(x2p2-m22)*k21)*(x1p2-m21)+((x1p2-m21)*k12+(x2p2-m22)*k22)*(x2p2-m22))+0.5*np.log(np.linalg.det(Sigma1)/np.linalg.det(Sigma2)) + np.log(P1/P2)

greska2=0
for i in h2:
   if i<0: #tamo gde nam je diskriminaciona prava za odbirke iz prve␣→klase veća od nule do znači da nam odbirci nisu dobro klasifikovani, jer za␣,→prvu klasu mora biti manja od nule
      greska2=greska2+1
fig = plt.figure()

plt.plot(y111, y211, '*m',label='Klasa I')
plt.plot(z111, z211, '.c',label='Klasa II')
plt.contour(x1,x2,y,0)


plt.grid(True)
plt.title('Prikaz dve klase u x1 x2 prostoru')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
fig.savefig('zadatak1-testKlasa.png')

Eps1=((greska1)/20.0) #verovatnoća greške prvog tipa (broj pogrešno klasifikovanih␣,→odbiraka iz prve klase kroz ukupan broj odbiraka)
Eps2=((greska2)/20.0) #verovatnoća greške drugog tipa (broj pogrešno␣,→klasifikovanih odbiraka iz druge klase kroz ukupan broj odbiraka)
P=0.5*Eps1+0.5*Eps2 #Ukupna Bajesova greska računa se na osnovu formule (8)
T=(1-P)*100 #Tačnost klasifikatora
print(T)
print(greska1,greska2)
####################################################TESTIRANJE ZA P(08/02) ##################
x1p2=y111
x2p2=y211
h28=0.5*(((x1p2-m11)*z11+(x2p2-m12)*z21)*(x1p2-m11)+((x1p2-m11)*z12+(x2p2-m12)*z22)*(x2p2-m12))-0.5*(((x1p2-m21)*k11+(x2p2-m22)*k21)*(x1p2-m21)+((x1p2-m21)*k12+(x2p2-m22)*k22)*(x2p2-m22))+0.5*np.log(np.linalg.det(Sigma1)/np.linalg.det(Sigma2)) 
greska281=0
for i in h28:
   if i>np.log(0.8/0.2): #tamo gde nam je diskriminaciona prava za odbirke iz prve␣→klase veća od nule do znači da nam odbirci nisu dobro klasifikovani, jer za␣,→prvu klasu mora biti manja od nule
      greska281=greska281+1


x1p2=z111
x2p2=z211
h28=0.5*(((x1p2-m11)*z11+(x2p2-m12)*z21)*(x1p2-m11)+((x1p2-m11)*z12+(x2p2-m12)*z22)*(x2p2-m12))-0.5*(((x1p2-m21)*k11+(x2p2-m22)*k21)*(x1p2-m21)+((x1p2-m21)*k12+(x2p2-m22)*k22)*(x2p2-m22))+0.5*np.log(np.linalg.det(Sigma1)/np.linalg.det(Sigma2))
greska282=0
for i in h28:
   if i<np.log(0.8/0.2): #tamo gde nam je diskriminaciona prava za odbirke iz prve␣→klase veća od nule do znači da nam odbirci nisu dobro klasifikovani, jer za␣,→prvu klasu mora biti manja od nule
      greska282=greska282+1
fig = plt.figure()

#plt.plot(y111, y211, '*m',label='Klasa I')
plt.plot(z111, z211, '+c',label='Klasa II')
plt.contour(x1,x2,l,0,colors='red')
plt.grid(True)
plt.title('Apriorne verovatnoce P1=0.2/ P2=0.8')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
fig.savefig('zadatak1-0208.png')
Eps1=((greska281)/20.0) #verovatnoća greške prvog tipa (broj pogrešno klasifikovanih␣,→odbiraka iz prve klase kroz ukupan broj odbiraka)
Eps2=((greska282)/20.0) #verovatnoća greške drugog tipa (broj pogrešno␣,→klasifikovanih odbiraka iz druge klase kroz ukupan broj odbiraka)
P=0.2*Eps1+0.8*Eps2 #Ukupna Bajesova greska računa se na osnovu formule (8)
T=(1-P)*100 #Tačnost klasifikatora
print(T)
print('greska 2/8')
print(greska281,greska282)
############################################### TESTIRANJE ZA P(02/08)  #################################################
x1p2=y111
x2p2=y211
h82=0.5*(((x1p2-m11)*z11+(x2p2-m12)*z21)*(x1p2-m11)+((x1p2-m11)*z12+(x2p2-m12)*z22)*(x2p2-m12))-0.5*(((x1p2-m21)*k11+(x2p2-m22)*k21)*(x1p2-m21)+((x1p2-m21)*k12+(x2p2-m22)*k22)*(x2p2-m22))+0.5*np.log(np.linalg.det(Sigma1)/np.linalg.det(Sigma2)) 
greska821=0
for i in h82:
   if i> np.log(0.2/0.8): #tamo gde nam je diskriminaciona prava za odbirke iz prve␣→klase veća od nule do znači da nam odbirci nisu dobro klasifikovani, jer za␣,→prvu klasu mora biti manja od nule
      greska821=greska821+1


x1p2=z111
x2p2=z211
h82=0.5*(((x1p2-m11)*z11+(x2p2-m12)*z21)*(x1p2-m11)+((x1p2-m11)*z12+(x2p2-m12)*z22)*(x2p2-m12))-0.5*(((x1p2-m21)*k11+(x2p2-m22)*k21)*(x1p2-m21)+((x1p2-m21)*k12+(x2p2-m22)*k22)*(x2p2-m22))+0.5*np.log(np.linalg.det(Sigma1)/np.linalg.det(Sigma2))
greska822=0
for i in h82:
   if i<np.log(0.2/0.8): 
      greska822=greska822+1
fig = plt.figure()

plt.plot(y111, y211, '*m',label='Klasa I')
plt.plot(z111, z211, '.c',label='Klasa II')
plt.contour(x1,x2,b,0,colors='blue')
plt.grid(True)
plt.title('Apriorne verovatnoce P1=0.8, P2=0.2')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
fig.savefig('zadatak1-0802.png')
Eps1=((greska821)/20.0) #verovatnoća greške prvog tipa (broj pogrešno klasifikovanih␣,→odbiraka iz prve klase kroz ukupan broj odbiraka)
Eps2=((greska822)/20.0) #verovatnoća greške drugog tipa (broj pogrešno␣,→klasifikovanih odbiraka iz druge klase kroz ukupan broj odbiraka)
P=0.8*Eps1+0.2*Eps2 #Ukupna Bajesova greska računa se na osnovu formule (8)
T=(1-P)*100 #Tačnost klasifikatora
print(T)
print('greska za 8/2')
print(greska821,greska822)
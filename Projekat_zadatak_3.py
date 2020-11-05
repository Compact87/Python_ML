import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split

data=pd.read_csv('UOAF1F2.csv')
U=data[data['klase']==1].iloc[:,0:2].values
O=data[data['klase']==2].iloc[:,0:2].values
A=data[data['klase']==3].iloc[:,0:2].values
####################################### 1.CRTANJE KLASA U 2D PROSTORU  #############################
fig = plt.figure()
plt.plot(U[:,0],U[:,1],'*m',label='Samoglasnik U')
plt.plot(O[:,0],O[:,1],'.y',label='Samoglasnik O')
plt.plot(A[:,0],A[:,1],'ob',label='Samoglasnik A')
plt.xlabel('F1')
plt.ylabel('F2')
plt.legend()
plt.title('Tri klase samoglasnika u 2D prostoru')
plt.show()
fig.savefig('zadatak1-Klase-2d.png')      

####################################### 2. Prikazati funkcije gustine verovatnoće klasa.
 ###################
x = np.arange(100,1500,0.1)
u1 = norm.pdf(x,np.mean(U[:,0]), np.std(U[:,0]))
u2 = norm.pdf(x,np.mean(U[:,1]), np.std(U[:,1]))
fig = plt.figure()
plt.plot(x,u1,'-b',label='U, prva')
plt.plot(x,u2,'-.k',label='U, druga')
plt.grid(True)
plt.legend(loc='upper right',shadow=True)
plt.title('Gustine raspodele za klasu U')
plt.show()
fig.savefig('zadatak3-gustineU.png')      

x = np.arange(200,1000,0.1)
o1 = norm.pdf(x,np.mean(O[:,0]), np.std(O[:,0]))
o2 = norm.pdf(x,np.mean(O[:,1]), np.std(O[:,1]))
fig = plt.figure()

plt.plot(x,o1,'-b',label='O prva')
plt.plot(x,o2,'-.b',label='O druga')
plt.grid(True)
plt.legend(loc='upper right',shadow=True)
plt.title('Gustine raspodele za klasu O')
plt.show()
fig.savefig('zadatak3-guystineO.png')      

x = np.arange(500,1500,0.1)
a1 = norm.pdf(x,np.mean(A[:,0]), np.std(A[:,0]))
a2 = norm.pdf(x,np.mean(A[:,1]), np.std(A[:,1]))
fig = plt.figure()

plt.plot(x,a1,'-b',label='A prva')
plt.plot(x,a2,'-.r',label='A druga')
plt.grid(True)
plt.legend(loc='upper right',shadow=True)
plt.title('Gustine raspodele za klasu A')
plt.show()
fig.savefig('zadatak3-gustineA.png')      


x = np.arange(100,1500,0.1)
u1 = norm.pdf(x,np.mean(U[:,0]), np.std(U[:,0]))
o1 = norm.pdf(x,np.mean(O[:,0]), np.std(O[:,0]))
a1 = norm.pdf(x,np.mean(A[:,0]), np.std(A[:,0]))
fig = plt.figure()
plt.plot(x,u1,'-b',label='U, prva')
plt.plot(x,o1,'-.y',label='O, prva')
plt.plot(x,a1,'--k',label='A prva')
plt.grid(True)
plt.legend(loc='upper right',shadow=True)
plt.title('Gustine raspodele prvo obelezje klasa A,O,U')
plt.show()
fig.savefig('zadatak3-gustineAOUprva.png') 

x = np.arange(100,1500,0.1)
u2 = norm.pdf(x,np.mean(U[:,1]), np.std(U[:,1]))
o2 = norm.pdf(x,np.mean(O[:,1]), np.std(O[:,1]))
a2 = norm.pdf(x,np.mean(A[:,1]), np.std(A[:,1]))
fig = plt.figure()
plt.plot(x,u2,'-b',label='U, druga')
plt.plot(x,o2,'-.y',label='O, druga')
plt.plot(x,a2,'--k',label='A druga')
plt.grid(True)
plt.legend(loc='upper right',shadow=True)
plt.title('Gustine raspodele drugo obelezje klasa A,O,U')
plt.show()
fig.savefig('zadatak3-gustineAOUdruga.png') 

########################################## 3. Podeliti skup na obučavajući i testirajući skup.
X=data.iloc[:,0:2].values
Y=data.iloc[:,2].values
X_obucavajuci, X_testirajuci, Y_obucavajuci, Y_testirajuci = train_test_split(X, Y, test_size=0.20, random_state=1)

from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import StratifiedKFold 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 

models = [] 
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_obucavajuci, Y_obucavajuci, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())) #štampamo srednju vrednost tačnosti
import seaborn as sns 
import warnings 
warnings.filterwarnings("ignore") 
fig = plt.figure()

ax = sns.boxplot(data=results) 
ax = sns.swarmplot(data=results, color="black")
plt.xticks(np.arange(0,	4),names)
plt.title('Por')
plt.show()


####4. Za izabrani klasifikator izvršiti evaluaciju modela na testirajućem skupu.#######
model=DecisionTreeClassifier()


from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score

model.fit(X_obucavajuci, Y_obucavajuci)
predictions = model.predict(X_testirajuci)
print(accuracy_score(Y_testirajuci, predictions))
print(confusion_matrix(Y_testirajuci, predictions))
print(classification_report(Y_testirajuci, predictions))

############Na istoj slici nacrtati podatke i granice odluke (klasifikacione linije)
############ i na obučavajućem skupu i na testirajućem skupu.
fig = plt.figure()

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_obucavajuci, Y_obucavajuci.astype(np.integer), clf=model) 


plot_decision_regions(X_testirajuci, Y_testirajuci.astype(np.integer), clf=model, legend=2)
plt.title('Granice odluke za testirajuci i obucavajuci skup')
plt.show()
fig.savefig('zadatak3-klasifLinije.png')      

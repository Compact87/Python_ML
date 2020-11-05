import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression #uvozimo linearnu regresiju
from sklearn import metrics #uvozimo metriku

data=pd.read_csv('Cenestanova.csv')
dataset=pd.read_csv('Cenestanova.csv')


#fig1, ax1 = plt.subplots()
#ax1.set_title('Basic Plot')
#ax1.boxplot(km)
#plt.show()
#np.mean(cene)#sr vrednost i mediajana su slicnih vrednosti pa je bezbedno zameniti null vrednosti srednjom vrednoscu
#print(np.median(km),np.mean(km))



data=data[['Cene stanova','Kvadratni metri']].replace(0, np.nan)
print('Nedostajuce vrednosti: %d' % np.isnan(data.values).sum())
print('Nedostajuce vrednosti u prvoj: %d' % np.isnan(data[['Cene stanova']].values).sum())
print('Nedostajuce vrednosti u drugoj: %d' % np.isnan(data[['Kvadratni metri']].values).sum())
#data.fillna(data.mean(), inplace=True)
#data=data.dropna()

#imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#t_skup = imputer.fit_transform(data.values)
print('Nedostajuce vrednosti: %d' % np.isnan(data.values).sum())
t_skup=data.values


####3. Prikazati rasuti dijagram podataka.
fig = plt.figure()

plt.scatter(t_skup[:,1],t_skup[:,0])
plt.xlabel('Kvadaratura')
plt.ylabel('Cene stanova')
plt.title('Vrednost stanova u zavisnosti od kvadrature')
plt.show()
fig.savefig('zadatak4scatter.png')      


df=pd.DataFrame({'cene': t_skup[:,0], 'Kvadratura': t_skup[:,1]})
df.corr()


####3. Izvršiti linearnu regresiju.
X1_obucavajuci, X1_testirajuci, Y1_obucavajuci, Y1_testirajuci = train_test_split(t_skup[:,1], t_skup[:,0], test_size=0.30, random_state=1)
regresija = LinearRegression() #za regresora biramo linearnu regresiju


X1_obucavajuci=np.reshape(X1_obucavajuci,(-1,1))
Y1_obucavajuci=np.reshape(Y1_obucavajuci,(-1,1))
X1_testirajuci=np.reshape(X1_testirajuci,(-1,1))
Y1_testirajuci=np.reshape(Y1_testirajuci,(-1,1))



regresija.fit(X1_obucavajuci, Y1_obucavajuci) 
print(regresija.coef_,regresija.intercept_)
fig = plt.figure()

plt.scatter(X1_obucavajuci,Y1_obucavajuci,color='red')
plt.plot(X1_obucavajuci, regresija.predict(X1_obucavajuci), color='blue')
plt.xlabel('Kvadratura')
plt.ylabel('Cene stanova')
plt.title('Vrednost stanova u zavisnosti od kvadrature')
plt.show()
fig.savefig('zadatak4obuka.png')      




Y_predikcija = regresija.predict(X1_testirajuci)


fig = plt.figure()
plt.scatter(X1_testirajuci,Y1_testirajuci,color='red')
plt.plot(X1_testirajuci, regresija.predict(X1_testirajuci), color='blue')
plt.xlabel('Kvadratura')
plt.ylabel('Cene stanova')
plt.title('Vrednost plate u zavisnosti od godina iskustva-testirajuci skup')
plt.show()
fig.savefig('zadatak4test.png')      




print(metrics.mean_squared_error(Y1_testirajuci/max(Y1_testirajuci),Y_predikcija/max(Y_predikcija)))
print(metrics.r2_score(Y1_testirajuci/max(Y1_testirajuci),Y_predikcija/max(Y_predikcija))) 


vrednost=100
vrednost=np.reshape(vrednost,(-1,1))
y_pred = regresija.predict(vrednost)
print('Vrednost stana od 100 kvadrata je: %.3f' % y_pred)

















































































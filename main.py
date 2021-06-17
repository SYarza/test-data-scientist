import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import statistics as sts

#Leer archivo en una tabla de datos
f=open('hotel.json',)
hotel_data=pd.read_json(f)

#Contar las visitas a los hoteles por país
countries=hotel_data['country']
country_counts=pd.Series(countries).value_counts()
months=hotel_data['arrival_date_month']
segments=hotel_data['market_segment']
is_canceled=hotel_data['is_canceled']

#Separando los países que proveen más huéspedes
ts=sum(country_counts)
i=0
s=0
while s/ts<0.90:
    s=s+country_counts[i]
    i=i+1
mcc=country_counts[0:i-1]
others=pd.Series([ts-s],index=['Others'])
mcco=mcc.append(others)

#Gráfica de los países que más visitan los hoteles
fig1=pd.Series(mcco).plot(kind='bar')
plt.show()

#Buscando outliers en la tarifa diaria promedio
adr=hotel_data['adr']

fig2=sns.boxplot(x=adr)
plt.show()

#Claramente, se ven dos outliers: un valor negativo y un
#valor muy separado de los cuantiles.
#Procederemos a borrarlos y a analizar otros posibles
#outliers por el metodo del interquantile range.
Q1=np.percentile(adr,25,interpolation='midpoint')
Q2=np.percentile(adr,50,interpolation='midpoint')
Q3=np.percentile(adr,75,interpolation='midpoint')
IQR=Q3-Q1
lim_inf=max(0,Q1-1.5*IQR)
lim_sup=Q3+1.5*IQR
hotel_datat=hotel_data
hotel_datat.drop(hotel_datat.index[(hotel_datat['adr']<0)],axis=0,inplace=True)
hotel_datat.drop(hotel_datat.index[(hotel_datat['adr']>4000)],axis=0,inplace=True)
adrt=hotel_datat['adr']
no=sum(adr>=lim_sup)+sum(adr<=lim_inf)
fig2=plt.plot(adr,'o')
plt.show()


swn=hotel_datat['stays_in_week_nights']
swen=hotel_datat['stays_in_weekend_nights']
st=swn+swen
tp=np.dot(adrt,st)
promedio=tp/sum(st)
print(promedio)

#Para ver cómo varían los precios a lo largo del
#año, calcularemos el promedio de la adr por semana
#del año.
weeks_adr=hotel_datat
for col in weeks_adr.columns:
    if not ('arrival_date_week_number' in col or 'adr' in col):
        del weeks_adr[col]
grouped=weeks_adr.groupby('arrival_date_week_number')
adr_mean=grouped.mean()
fig3=plt.plot(adr_mean)
plt.show()

#Graficamos los meses por la cantidad de
#reservaciones de mayor a menos ocupación
months_counts=pd.Series(months).value_counts()
fig4=pd.Series(months_counts).plot(kind='bar')
plt.show()

#Graficamos la cantidad de noches por
#reservación y la frecuencia de cada cantidad
#Dejamos fuera el aquéllas reservaciones mayores
#a 14 días, que son menos del 1% del total de
#reservaciones
n=0
st2=st[st>0]
while sum(st2<=n)/len(st2)<0.99:
    n=n+1
stn=st2[st2<=n]
fig5=stn.hist(bins=n)
plt.show()

#Número de reservaciones por segmento de mercado
segments_counts=pd.Series(segments).value_counts()
fig6=pd.Series(segments_counts).plot(kind='bar')
plt.show()

#Calculamos el número de reservaciones canceladas
nrc=sum(is_canceled)
print(nrc)

#Graficamos el número de cancelaciones por mes
months_canceled=months[is_canceled==1]
months_canceled_counts=pd.Series(months_canceled).value_counts()
fig7=pd.Series(months_canceled_counts).plot(kind='bar')
plt.show()

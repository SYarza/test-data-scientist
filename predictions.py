import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

#Leyendo archivo y guárdandolo en una base de
#datos.
f=open('hotel.json',)
hotel_data=pd.read_json(f)

#Calculando la proporción de valores faltantes en
#los datos categóricos y convirtiéndolos en NaN.
for col in hotel_data.columns:
    if hotel_data[col].dtype == 'object':
        hotel_data.loc[hotel_data[col].str.contains('NULL'), col] = np.nan
        hotel_data.loc[hotel_data[col].str.contains('Undefined', na=False), col] = np.nan
null_series = hotel_data.isnull().mean()
print(null_series[null_series > 0])

#Desechando los valores NaN en las variables en
#las que la cantidad de valores faltantes es menor
#del 1%
subset = [
    'meal',
    'country',
    'children',
    'market_segment',
    'distribution_channel'
]
hotel_data = hotel_data.dropna(subset=subset)

#Ahora, rellenamos los valores faltantes en las
#otras variables con algún número aleatorio
hotel_data.loc[hotel_data.agent.isnull(), 'agent'] = '702'
hotel_data.loc[hotel_data.company.isnull(), 'company'] = '702'

#Borramos aquéllos datos en los que la tarifa
#diaria promedio no es positiva
hotel_data = hotel_data[hotel_data.adr > 0]

#En los datos con valores enteros o flotantes,
#removemos outliers con el método de IQR
hotel_data['children']=pd.to_numeric(hotel_data['children'])
cleaned = hotel_data.copy()

columns = [
    'lead_time',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'babies',
    'adr',
]

for col in columns:
    q1 = hotel_data[col].quantile(0.25)
    q3 = hotel_data[col].quantile(0.75)

    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    print(f'Lower point: {round(lower, 2)} \t upper point: {round(upper, 2)} \t {col}')

    if lower == upper:
        continue

    cond1 = (cleaned[col] >= lower) & (cleaned[col] <= upper)
    cond2 = cleaned[col].isnull()
    cleaned = cleaned[cond1 | cond2]

# create new features: total price and total nights
cleaned.loc[:, 'total_nights'] = \
cleaned['stays_in_week_nights'] + cleaned['stays_in_weekend_nights']
cleaned.loc[:, 'price'] = cleaned['adr'] * cleaned['total_nights']
# create numpy array
X = np.array(cleaned[['total_nights', 'price']])
# create model
ee = EllipticEnvelope(contamination=.01, random_state=0)
# predictions
y_pred_ee = ee.fit_predict(X)
# predictions (-1: outlier, 1: normal)
anomalies = X[y_pred_ee == -1]
# plot data and outliers
plt.figure(figsize=(15, 8))
plt.scatter(X[:, 0], X[:, 1], c='white', s=20, edgecolor='k')
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red');
plt.show()

hotel_data_cleaned = cleaned[y_pred_ee != -1].copy()

#Separamos los datos de los dos hoteles
urban_data=hotel_data_cleaned[hotel_data_cleaned['hotel']!='Resort Hotel'].copy()
resort_data=hotel_data_cleaned[hotel_data_cleaned['hotel']=='Resort Hotel'].copy()
print(len(urban_data))
print(len(resort_data))
del urban_data['hotel']
del resort_data['hotel']

hotel_data_le = urban_data.copy()
le = LabelEncoder()

categoricals = [
    'arrival_date_month',
    'meal',
    'country',
    'market_segment',
    'distribution_channel',
    'reserved_room_type',
    'assigned_room_type',
    'deposit_type',
    'agent',
    'company',
    'customer_type',
    'reservation_status',
    'reservation_status_date'
]

for col in categoricals:
    hotel_data_le[col] = le.fit_transform(hotel_data_le[col])
plt.figure(figsize=(40, 30))
sns.heatmap(hotel_data_le.corr(), annot=True, fmt='.2f');
plt.show()

for col in categoricals:
    print(col)
    print(len(pd.unique(urban_data[col])))

columns = [
    'reservation_status',
    'reservation_status_date'
]

urban_data = urban_data.drop(columns, axis=1)
resort_data = resort_data.drop(columns, axis=1)
hotel_data_le = hotel_data_le.drop(columns, axis=1)

new_categoricals = [col for col in categoricals if col in urban_data.columns]
hotel_data_hot = pd.get_dummies(data=urban_data, columns=new_categoricals)
resort_data_hot = pd.get_dummies(data=resort_data, columns=new_categoricals)
X_hot = hotel_data_hot.drop('is_canceled', axis=1)
X_le = hotel_data_le.drop('is_canceled', axis=1)
y = urban_data['is_canceled']

X_urban, X_resort, y_urban, y_resort = train_test_split(X_hot, y, random_state=702)

lr=LogisticRegression(solver='saga')
log = lr.fit(X_urban, y_urban)
y_pred = log.predict(X_resort)
print(accuracy_score(y_resort, y_pred))
print(classification_report(y_resort, y_pred))

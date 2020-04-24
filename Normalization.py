import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import csv
import math

# Funzione per il calcolo del valore assoluto dei numeri complessi


def absolute_value_complex_arr(parameter):
    auxC = []
    for j in parameter:
        j = j.replace('(', '')
        j = j.replace(')', '')
        compl = complex(j)
        auxC.append(compl)

    real_part = list(np.real(auxC))
    imagin_part = list(np.imag(auxC))

    result = []
    for j in range(0, len(real_part)):
        res = math.sqrt(pow(real_part[j], 2) + pow(imagin_part[j], 2))
        result.extend([res])

    return np.asarray(result)


data = pd.read_csv('features.csv', sep=",", header=0, low_memory=False)
clm1 = data.columns.values

# Eliminazione delle righe problematiche
data = data.drop([1582], axis=0)
data = data.drop([2644], axis=0)


cls = data.Class
idS = data.IDSubject

x = data.drop('Class', axis=1)
x = x.drop('IDSubject', axis=1)

clm2 = x.columns.values
index = []

# Creazione matrice contenente i dati in un unico formato
matrix1 = []
for i in clm2:
    con = 0
    if "pks" not in i:
        values = x[i].values
    else:
        values = absolute_value_complex_arr(x[i].values)
    values = values.reshape((len(values), 1))
    matrix1.append(values)

matrix2 = []
for i in range(0, len(matrix1[0])):
    rowStamp = []
    for k in range(0, len(matrix1)):
        app = matrix1[k]
        rowStamp.extend(app[i])
    matrix2.append(rowStamp)

for k in range(0, len(matrix2)):
    for i in range(0, len(matrix2[0])):
        if isinstance(matrix2[k][i], str):
            matrix2[k][i] = float(matrix2[k][i])

# Normalizzazione dei dati
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(matrix2)
matrixFin = scaler.transform(matrix2)

# Moltiplicazione dei dati per 10000
for i in range(0, len(matrixFin)):
    for k in range(0, len(matrixFin[0])):
        matrixFin[i][k] = matrixFin[i][k]*10000

# Creazione del dataset Normalizzato
with open('normalizedDataset.csv', 'w', newline='') as f:
    wtr = csv.writer(f)
    wtr.writerow(clm1)

    for i in range(0, len(matrix2)):
        rowStamp = [idS.iloc[i]]
        rowStamp.extend(matrixFin[i])
        rowStamp.append(cls.iloc[i])
        wtr.writerow(rowStamp)

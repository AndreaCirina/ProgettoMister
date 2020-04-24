import pandas as pd
from numpy import mean, var, fft, diff, sign, correlate
from scipy.stats import kurtosis, skew
import csv

# Creazione nome delle feature

stringSens = "Acc Gyr Mag".split()
stringAx = "x y z".split()
stringFeature = "max min avg var skew kurt aut pks fpk".split()

clm = ["IDSubject"]
for i in range(5):
    for s in range(3):
        for a in range(3):
            for f in range(9):

                if stringFeature[f] == "aut":
                    for u in range(1, 11):
                        clm.append(str(i) + "-" + stringSens[s] + "-" + stringAx[a] + "-" + stringFeature[f] + str(u))
                else:
                    if stringFeature[f] == "pks" or stringFeature[f] == "fpk":
                        for u in range(1, 6):
                            clm.append(
                                str(i) + "-" + stringSens[s] + "-" + stringAx[a] + "-" + stringFeature[f] + str(u))
                    else:
                        clm.append(str(i) + "-" + stringSens[s] + "-" + stringAx[a] + "-" + stringFeature[f])

clm.append("Class")


# Calcolo delle feature e scrittura sul file
stringPath = "D:\\File\\Documenti\\Universita\\2019-2020\\2_semestre\\Tirocinio\\FileVari\\Dataset\\data"
with open('features.csv', 'w', newline='') as f:
    wtr = csv.writer(f)
    wtr.writerow(clm)
    for p in range(1, 9):
        for a in range(1, 20):
            auxStrA = ""
            if a < 10:
                auxStrA = "0"

            auxStrA += str(a)
            for s in range(1, 61):
                auxStr = ""
                if s < 10:
                    auxStr = "0"

                auxStr += str(s)
                stringPathFile = stringPath + "\\a" + auxStrA + "\\p" + str(p) + "\\s" + auxStr + ".txt"

                X = pd.read_csv(stringPathFile, sep=",", header=None)
                row = ["P" + str(p)]
                for i in range(X.shape[1]):
                    aux = []
                    # Valore massimo
                    aux.append(max(X[i]))

                    # Valore minimo
                    aux.append(min(X[i]))

                    # Valore medio
                    aux.append(mean(X[i]))

                    # Varianza
                    aux.append(var(X[i], ddof=1))

                    # Assimetria
                    aux.append(skew(X[i]))

                    # Curtosi
                    aux.append(kurtosis(X[i]))

                    # Autocorrelazione
                    acf = correlate(X[i], X[i], 'full')[-len(X[0]):]
                    aut = []
                    for j in range(acf.size):
                        if j % 14 == 0:
                            aut.append(acf[j])

                    aut.append(acf[124])
                    aux.extend(aut)

                    # Trasformata discreta di Fourier
                    fourier = (fft.rfft(X[i] - X[i].mean()))
                    freq = fft.rfftfreq(X[i].size, d=1. / 25)
                    inflection = diff(sign(diff(fourier)))
                    peaks = (inflection < 0).nonzero()[0] + 1

                    # primi 5 picchi trasformata di fourier 
                    peak = fourier[peaks]
                    
                    # Frequenza dei picchi della trasformata discreta di fourier 
                    signal_freq = freq[peaks]
                    aux.extend(peak[:5])
                    aux.extend(signal_freq[:5])
                    row.extend(aux)

                row.extend(["a" + str(a)])
                wtr.writerow(row)

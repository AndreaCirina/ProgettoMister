import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import psycopg2

data = pd.read_csv('normalizedDataset.csv', sep=",", header=0, low_memory=False)
classi = "a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19".split()

# Suddivisione dataset per test e train
X_train = data[data.IDSubject != "P8"]
X_test = data[data.IDSubject == "P8"]

X_train = X_train.drop('IDSubject', axis=1)
y_train = X_train.Class
X_train = X_train.drop('Class', axis=1)

X_test = X_test.drop('IDSubject', axis=1)
y_test = X_test.Class
X_test = X_test.drop('Class', axis=1)

# Calcolo feature più significative
data = data.drop('IDSubject', axis=1)
X_data = data.drop('Class', axis=1)
y_data = data.Class
test = SelectKBest(score_func=chi2, k=30)
fit = test.fit(X_data, y_data)

# Estrazione feature
np.set_printoptions(precision=3)
X_train = fit.transform(X_train)
X_test = fit.transform(X_test)

# Classificazione
clf = RandomForestClassifier(random_state=None)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Calcolo accuracy, precision, recall e F-Measure
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
arr = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print("Precision: ", arr[0], "Recall: ", arr[1], "F-Measure:", arr[2])
C = confusion_matrix(y_pred, y_test)
df_cm = pd.DataFrame(C, classi, classi)
sns.set(font_scale=1.4)
graphic = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="YlGnBu", fmt='g')
plt.show()

#connessione database
con = psycopg2.connect(
    host="localhost",
    database="thefooturelab",
    user="postgres",
    password="postgres"
)

cur = con.cursor()

# Inserimento delle predizioni nel database
i = 0
for h in y_pred:
    print(h)
    cur.execute("insert into public.events (id, start_event, end_event, note, match_id, period_id,player_id, team_id) "
                "values (%s,%s,%s,%s,%s,%s,%s,%s)", (i, i, i+1, h, 11, 0, 1, 1))
    cur.execute("insert into public.\"events_tagCombinations\" (id,event_id, tagcombinationtoversion_id,sort_value)"
                " values (%s,%s,%s,%s) ", (i, i, int(h[1:]), i))
    i += 1

con.commit()
con.close()
cur.close()

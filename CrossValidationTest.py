import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Load the Diabetes dataset
data = pd.read_csv('C:\\Users\\andre\\Desktop\\COSAR-DS-1.0\\test_trainingA.csv')
y = data.classes
df = data.drop('classes', axis=1)

classi = "brushingTeeth climbingUp climbingDown ridingBycicle jogging standingStill strolling walkingDownstairs " \
         "walkingUpstairs writingOnBlackboard".split()

# fit a model (Random Forest)
clf = RandomForestClassifier(random_state=None)

# Perform 4-fold cross validation
kf = KFold(n_splits=4, shuffle=True, random_state=0)
scores = cross_val_score(clf, df, y, cv=kf)
predictions = cross_val_predict(clf, df, y, cv=kf)
arr = precision_recall_fscore_support(y, predictions, average='weighted')

print("Cross-validated scores:", scores)
print('>> Mean CV score is: ', round(np.mean(scores), 3))

# Confusion Matrix
C = confusion_matrix(predictions, y)
df_cm = pd.DataFrame(C, classi, classi)
sns.set(font_scale=1.4)
graphic = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="YlGnBu", fmt='g')
plt.show(graphic)

print("Precision: ", arr[0], "Recall: ", arr[1], "F-1:", arr[2])

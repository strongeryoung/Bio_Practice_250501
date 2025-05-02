from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

y = [2, 3, 4, 1, 5]
y_pred = [1.5, 2.5, 5, 2, 4.5]

mae = mean_absolute_error(y, y_pred)
print("mae :", mae)

mse = mean_absolute_error(y, y_pred)
print("mse :", mse)

r2_score = r2_score(y, y_pred)
print("r2 score :", r2_score)

# Classification 에서 평가 척도
# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y = [1, 0, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0]

cm = confusion_matrix(y, y_pred)
print("confusion matrix:", cm)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


## Accuracy 구하기
# scikit-learn 라이브러리에서 accuracy_score 불러오기

y = [1, 0, 1, 0, 1]
y_pred = [1, 1, 1, 0, 0]

acc = accuracy_score(y, y_pred)

print("accuracy :", acc)

# Precision
from sklearn.metrics import precision_score

y = [1, 0, 1, 0, 1]
y_pred = [1, 1, 1, 0, 0]

precision = precision_score(y, y_pred)

print("precision :", precision)


# Recall
from sklearn.metrics import recall_score
y = [1, 0, 1, 0, 1]
y_pred = [1, 1, 1, 0, 0]
recall = recall_score(y, y_pred)
print("recall :", recall)


# f1 score
from sklearn.metrics import f1_score

y = [1, 0, 1, 0, 1]
y_pred = [1, 1, 1, 0, 0]

f1 = f1_score(y, y_pred)
print("f1 score :", f1)


## ROC 곡선 그리기

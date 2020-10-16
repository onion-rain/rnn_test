from sklearn.metrics import precision_score, recall_score
y_true = [1,0,1]
y_pred=[0,1,1]
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
score = (precision * recall) / (0.4 * precision + 0.6 * recall)
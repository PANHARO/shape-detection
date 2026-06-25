from sklearn.metrics import accuracy_score, classification_report
from features import X_test, y_test
from training import dt, rf, svm

dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)
svm_pred = svm.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)
svm_acc = accuracy_score(y_test, svm_pred)

print(f"Decision Tree accuracy is: {round(dt_acc * 100, 2)}%")
print(f"Random Forest accuracy is: {round(rf_acc * 100, 2)}%")
print(f"SVM accuracy is: {round(svm_acc * 100, 2)}%")

print(classification_report(y_test, rf_pred))

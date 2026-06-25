from sklearn.metrics import accuracy_score, classification_report
from features_updated import X_test, y_test
from training import dt, rf, svm

#Prediction
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)
svm_pred = svm.predict(X_test)

#Calculating accuracy
dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)
svm_acc = accuracy_score(y_test, svm_pred)

print(f"Decision Tree accuracy is: {round(dt_acc * 100, 2)}%")
print(f"Random Forest accuracy is: {round(rf_acc * 100, 2)}%")
print(f"SVM accuracy is: {round(svm_acc * 100, 2)}%")

#Report
print("Classification Report(Random Forest):")
print(classification_report(y_test, rf_pred))
print("Classification Report(Decision Tree):")
print(classification_report(y_test, dt_pred))
print("Classification Report(SVM):")
print(classification_report(y_test, svm_pred))

# Extra discussion
best = max([("Decision Tree", dt_acc), ("Random Forest", rf_acc), ("SVM", svm_acc)], key=lambda x: x[1])
worst = min([("Decision Tree", dt_acc), ("Random Forest", rf_acc), ("SVM", svm_acc)], key=lambda x: x[1])
print(f"The best model is {best[0]} with {round(best[1] * 100, 2)}%")
print(f"The worst model is {worst[0]} with {round(worst[1] * 100, 2)}%")
print('The tree-based models that we use such as Decision Tree and Random Forest can perform well because the extracted features, such as area, perimeter, circularity, aspect ratio, vertices, and Hu moments, provide useful rules for separating shape classes.')
print('SVM is more sensitive to the scale of the input features, so feature scaling is important because large values such as area and perimeter can otherwise dominate smaller features such as circularity and Hu moments.')


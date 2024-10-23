import sklearn
from sklearn.metrics import confusion_matrix, classification_report


def test_model(model, x_test, y_test):
    # Predict the labels for the test set
    y_pred = model.predict(x_test)

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print the confusion matrix
    print('Confusion Matrix:')
    print(conf_matrix)

    # Generate the classification report
    class_report = classification_report(y_test, y_pred)

    # Print the classification report
    print('Classification Report:')
    print(class_report)

    return conf_matrix, class_report
import sklearn
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def test_model(model, x_test, y_test, verbose = False, save_file_path = None):
    # Predict the labels for the test set
    y_pred = model.predict(x_test)

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Generate the classification report
    class_report = classification_report(y_test, y_pred)

    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    # Compute precision, recall, and F1 scores for each class
    precision = sklearn.metrics.precision_score(y_test, y_pred, average=None)
    recall = sklearn.metrics.recall_score(y_test, y_pred, average=None)
    f1_scores = sklearn.metrics.f1_score(y_test, y_pred, average=None)

    # Compute the averages, minimums of precision, recall, and F1 scores
    avg_precision = precision.mean()
    avg_recall = recall.mean()
    avg_f1_score = f1_scores.mean()

    min_precision = precision.min()
    min_recall = recall.min()
    min_f1_score = f1_scores.min()

    # Compute class accuracies
    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # Store the results in a dictionary
    metrics = {
        'accuracy' : accuracy,
        'precision': precision,
        'recall': recall,
        'f1_scores': f1_scores,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1_score': avg_f1_score,
        'min_precision': min_precision,
        'min_recall': min_recall,
        'min_f1_score': min_f1_score,
        'class_accuracies': class_accuracies
    }
    if verbose:
        # Print the confusion matrix
        print('Confusion Matrix:')
        print(conf_matrix)
        
        # Print the accuracy
        print('Accuracy:', round(accuracy, 2))
        
        # Print the precision, recall, and F1 scores for each class
        for i, (p, r, f1) in enumerate(zip(precision, recall, f1_scores)):
            print(f'Class {i}:')
            print(f'  Precision: {round(p, 2)}')
            print(f'  Recall: {round(r, 2)}')
            print(f'  F1 Score: {round(f1, 2)}')
        
        # Print the average and minimum precision, recall, and F1 scores
        print('Average Precision:', round(avg_precision, 2))
        print('Average Recall:', round(avg_recall, 2))
        print('Average F1 Score:', round(avg_f1_score, 2))
        print('Minimum Precision:', round(min_precision, 2))
        print('Minimum Recall:', round(min_recall, 2))
        print('Minimum F1 Score:', round(min_f1_score, 2))
        
        # Print class accuracies
        for i, acc in enumerate(class_accuracies):
            print(f'Class {i} Accuracy: {round(acc, 2)}')

        # Save the metrics to csv file
        if save_file_path is not None:
            df = pd.DataFrame(columns=['class', 'precision', 'recall', 'f1', 'accuracy'])
            for i in range(len(precision)):
                df = df.append({'class': str(i), 'precision': precision[i], 'recall': recall[i], 'f1': f1_scores[i], 'accuracy': class_accuracies[i]}, ignore_index=True)
            df = df.append({'class': 'average', 'precision': avg_precision, 'recall': avg_recall, 'f1': avg_f1_score, 'accuracy': accuracy}, ignore_index=True)
            df.to_csv(save_file_path, index=False)

    return metrics
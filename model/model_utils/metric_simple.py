import numpy as np


def calculate_metrics(pred, label):
    pred = pred.flatten()
    label = label.flatten()

    tp = np.sum((pred == 1) & (label == 1))  # True Positives
    fp = np.sum((pred == 1) & (label == 0))  # False Positives
    fn = np.sum((pred == 0) & (label == 1))  # False Negatives
    tn = np.sum((pred == 0) & (label == 0))  # True Negatives

    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)  # Recall (also PD, Positive Detection Rate)

    if (tp + fp) == 0:
        precision = 0  # Precision, to avoid confusion with map in multi-class, we'll call it precision here
    else:
        precision = tp / (tp + fp)

    map_ = precision

    if (fp + tn) == 0:
        fa = 0  # False Alarm Rate, to avoid division by zero
    else:
        fa = fp / (fp + tn)

    return map_, recall, recall, fa  # Recall and PD are the same

pred = np.array([1, 0, 1, 1, 0, 0, 1, 0])
label = np.array([1, 0, 1, 0, 0, 1, 1, 0])

map_, recall, pd, fa = calculate_metrics(pred, label)
print(f"Mean Average Precision (as Precision in binary): {map_}")
print(f"Recall (PD): {recall}")
print(f"False Alarm Rate: {fa}")
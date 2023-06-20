import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn import svm
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

def membership_inference(svm_model, x_train, y_train, x_test, y_test, n_shadow_models=2):
    n_train = len(x_train)
    n_test = len(x_test)

    x_combined = np.concatenate([x_train, x_test])
    y_combined = np.concatenate([y_train, np.zeros_like(y_test)])  # Label the test data as 0

    shadow_scores = []
    for _ in range(n_shadow_models):
        shadow_indices = np.random.choice(len(x_combined), n_train, replace=False)
        shadow_model = svm.SVC(gamma='scale', probability=True)
        shadow_model.fit(x_combined[shadow_indices], y_combined[shadow_indices])
        shadow_scores.append(shadow_model.predict_proba(x_combined)[:, 1])  # keep only positive class scores

    inferred_scores = np.stack(shadow_scores).mean(axis=0)
    actual_membership = np.concatenate([np.ones(n_train), np.zeros(n_test)])

    return actual_membership, inferred_scores

def plot_roc_auc(actual_membership, inferred_scores, label):
    fpr, tpr, _ = roc_curve(actual_membership, inferred_scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{label} ROC curve (area = {roc_auc:.2f})')

def poison_data(y_train, poison_fraction):
    poison_indices = np.random.choice(len(y_train), int(len(y_train) * poison_fraction), replace=False)
    y_train_poisoned = y_train.copy()
    y_train_poisoned[poison_indices] = (y_train[poison_indices] + 5) % 10
    return y_train_poisoned

indices = np.random.choice(len(x_train), 8000, replace=False)

avg_svs = []
accs = []
plt.figure(figsize=(10, 10))

for poison_fraction in [0.01, 0.05, 0.1, 0.2]:
    svm_model = svm.SVC(gamma='scale')
    svm_model.fit(x_train[indices], y_train[indices])

    actual_membership, inferred_membership_before = membership_inference(svm_model, x_train[indices], y_train[indices], x_test, y_test)
    plot_roc_auc(actual_membership, inferred_membership_before, f'Before Poisoning ({poison_fraction*100:.0f}%)')
    acc_before = accuracy_score(actual_membership, inferred_membership_before.round())
    
    avg_svs_before = np.mean(svm_model.n_support_)
    
    y_train_poisoned = poison_data(y_train[indices], poison_fraction)
    svm_model.fit(x_train[indices], y_train_poisoned)

    actual_membership, inferred_membership_after = membership_inference(svm_model, x_train[indices], y_train_poisoned, x_test, y_test)
    plot_roc_auc(actual_membership, inferred_membership_after, f'After Poisoning ({poison_fraction*100:.0f}%)')
    acc_after = accuracy_score(actual_membership, inferred_membership_after.round())
    
    avg_svs_after = np.mean(svm_model.n_support_)
    
    avg_svs.append([avg_svs_before, avg_svs_after])
    accs.append([acc_before, acc_after])

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Different Poisoning Amounts')
plt.legend(loc="lower right")
plt.show()

categories = ['Before Poisoning', 'After Poisoning']
poison_labels = ['1%', '5%', '10%', '20%']

plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(1, 4, i+1)
    sns.barplot(x=categories, y=avg_svs[i])
    plt.title('Avg Support Vectors for ' + poison_labels[i])
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(1, 4, i+1)
    sns.barplot(x=categories, y=accs[i])
    plt.title('Accuracy for ' + poison_labels[i])
plt.tight_layout()
plt.show()



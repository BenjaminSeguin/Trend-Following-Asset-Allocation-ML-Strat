import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import os


def plot_mean_importance(
    asset_class: str,
    aggregated_importances: List[np.ndarray],
    feature_names: List[str],
    model_name: str,
):
    """
    Plot mean feature importances for a given asset class and save the plot.

    Parameters:
    ----------
    asset_class : str
        The name of the asset class.
    aggregated_importances : list
        List of feature importance arrays for the asset class.
    feature_names : list
        List of feature names.
    model_name : str
        Name of the model being evaluated (e.g., "Ridge Regression", "Random Forest").
    """
    # Create Outputs folder if it doesn't exist
    output_folder = "Outputs"
    os.makedirs(output_folder, exist_ok=True)

    if aggregated_importances:
        # Stack importances and compute mean
        importances_array = np.vstack(aggregated_importances)
        mean_importances = np.mean(importances_array, axis=0)

        # Create DataFrame
        mean_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean Importance': mean_importances
        }).sort_values(by='Mean Importance', ascending=False)

        print(f"\nAverage Feature Importances for {asset_class} ({model_name}):")
        print(mean_importance_df)

        # Plot the mean feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(mean_importance_df['Feature'], mean_importance_df['Mean Importance'])
        plt.gca().invert_yaxis()  # Most important at the top
        plt.xlabel('Mean Importance')
        plt.title(f'Average Feature Importances for {asset_class} ({model_name})')

        # Save plot
        filename = f"{asset_class}_{model_name}_importance.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        print(f"Plot saved to: {filepath}")
    else:
        print(f"No feature importances collected for {asset_class} ({model_name}).")


def display_average_r2(
    asset_class: str,
    r2_scores: List[float],
    model_name: str,
):
    """
    Display the average R² score for a given asset class.

    Parameters:
    ----------
    asset_class : str
        The name of the asset class.
    r2_scores : list
        List of R² scores for the asset class.
    model_name : str
        Name of the model being evaluated.
    """
    if r2_scores:
        avg_r2 = np.mean(r2_scores)
        print(f"Average R² for {model_name} in {asset_class}: {avg_r2:.4f}")
    else:
        print(f"No R² scores collected for {model_name} in {asset_class}.")

def compute_classification_metrics(
    asset_class: str,
    confusion_matrices: List[np.ndarray]
) -> Dict[str, float]:
    """
    Compute average precision, recall, accuracy, and F1 score for a given asset class.

    Parameters:
    ----------
    asset_class : str
        The name of the asset class.
    confusion_matrices : list
        List of confusion matrices for the asset class.

    Returns:
    -------
    dict
        Dictionary containing average precision, recall, accuracy, and F1 score.
    """
    recall_scores = []
    precision_scores = []
    accuracy_scores = []
    f1_scores = []

    for cm in confusion_matrices:
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tn = cm[0, 0]

        # Calculate Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_scores.append(precision)

        # Calculate Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_scores.append(recall)

        # Calculate Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracy_scores.append(accuracy)

        # Calculate F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    # Compute average metrics
    mean_metrics = {
        'Precision': np.mean(precision_scores),
        'Recall': np.mean(recall_scores),
        'Accuracy': np.mean(accuracy_scores),
        'F1 Score': np.mean(f1_scores)
    }

    # Display the results
    print(f"\nMetrics for {asset_class}:")
    for metric, value in mean_metrics.items():
        print(f"Average {metric}: {value:.2f}")

    return mean_metrics


def plot_average_confusion_matrix(
    asset_class: str,
    confusion_matrices: List[np.ndarray]
):
    """
    Plot the average confusion matrix for a given asset class and save the plot.

    Parameters:
    ----------
    asset_class : str
        The name of the asset class.
    confusion_matrices : list
        List of confusion matrices for the asset class.
    """

    # Create Outputs folder if it doesn't exist
    output_folder = "Outputs"
    os.makedirs(output_folder, exist_ok=True)

    if confusion_matrices:
        # Sum all confusion matrices and compute the mean
        summed_cm = np.sum(confusion_matrices, axis=0)
        mean_cm = summed_cm / len(confusion_matrices)

        # Normalize to show percentages
        cm_normalized = mean_cm.astype('float') / mean_cm.sum(axis=1)[:, np.newaxis]

        print(f"\nAverage Confusion Matrix for {asset_class}:")
        print(cm_normalized)

        # Visualize the average confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(
            cm_normalized * 100, 
            annot=True, 
            fmt=".2f", 
            cmap="Blues", 
            xticklabels=[0, 1], 
            yticklabels=[0, 1]
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Average Confusion Matrix for {asset_class} (in percentage)')

        # Save plot
        filename = f"{asset_class}_average_confusion_matrix.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        print(f"Plot saved to: {filepath}")
    else:
        print(f"No confusion matrices collected for {asset_class}.")

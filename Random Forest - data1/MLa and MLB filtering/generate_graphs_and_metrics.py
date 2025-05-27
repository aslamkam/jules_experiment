import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        'R2': round(r2, 4),
        'MSE': round(mse, 4),
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4)
    }

def save_metrics(metrics, filename):
    """Save metrics to a text file."""
    with open(filename, 'w') as f:
        f.write("# Model Performance Metrics\n\n")
        for metric, value in metrics.items():
            f.write(f"* {metric}: {value}\n")

def create_parity_plot(y_true, y_pred, title, filename):
    """Create and save a parity plot."""
    plt.figure(figsize=(8, 8))
    
    # Calculate axis limits
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    
    # Add some padding to the limits
    padding = (max_val - min_val) * 0.1
    plt_min = min_val - padding
    plt_max = max_val + padding
    
    # Create the scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add the reference line (45 degrees)
    plt.plot([plt_min, plt_max], [plt_min, plt_max], 'r--', label='y=x')
    
    # Set equal scaling and limits
    plt.axis('square')
    plt.xlim(plt_min, plt_max)
    plt.ylim(plt_min, plt_max)
    
    # Labels and title
    plt.xlabel('Experimental pKa')
    plt.ylabel('Predicted pKa')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Read the CSV file
    try:
        df = pd.read_csv('MLa_all_molecules.csv')
        
        # Calculate and save metrics for MLa
        mla_metrics = calculate_metrics(df['pka_value'], df['MLa_Prediction'])
        save_metrics(mla_metrics, 'MLa_Metrics.txt')

        df = pd.read_csv('MLb_all_molecules.csv')
        # Calculate and save metrics for MLb
        mlb_metrics = calculate_metrics(df['pka_value'], df['MLb_Prediction'])
        save_metrics(mlb_metrics, 'MLb_Metrics.txt')
        
        # Create parity plots
        create_parity_plot(
            df['pka_value'], 
            df['MLa_Prediction'], 
            'MLa Parity Plot', 
            'MLa_Parity_Plot.png'
        )
        
        create_parity_plot(
            df['pka_value'], 
            df['MLb_Prediction'], 
            'MLb Parity Plot', 
            'MLb_Parity_Plot.png'
        )
        
        print("Analysis completed successfully!")
        print("\nMLa Metrics:", mla_metrics)
        print("MLb Metrics:", mlb_metrics)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
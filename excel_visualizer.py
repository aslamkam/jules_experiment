import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure the output directory exists
os.makedirs('infographics', exist_ok=True)

excel_file = pd.ExcelFile("Model_Results_Tabulation.xlsx")
sheet_names = excel_file.sheet_names

for sheet_name in sheet_names:
    df = excel_file.parse(sheet_name)

    # Clean up sheet name for filename
    safe_sheet_name = "".join(c if c.isalnum() else "_" for c in sheet_name)

    plt.figure(figsize=(12, 8))
    # Check if 'Folder' and 'Test R2' columns exist
    if 'Folder' in df.columns and 'Test R2' in df.columns:
        # Sort by Test R2 for better visualization, handle NaNs by dropping them for plotting
        df_sorted = df.dropna(subset=['Test R2']).sort_values(by='Test R2', ascending=False)

        if not df_sorted.empty: # Proceed only if there's data to plot
            bars = plt.bar(df_sorted['Folder'], df_sorted['Test R2'], color='skyblue')

            plt.xlabel("Model (Folder)")
            plt.ylabel("Test R2 Score")
            plt.title(f"Model Comparison by Test R2 Score - {sheet_name}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout() # Adjust layout to prevent labels from being cut off

            # Add R2 values on top of the bars
            for bar in bars:
                yval = bar.get_height()
                # Check if yval is a float or can be converted to float, otherwise skip annotation
                try:
                    yval_float = float(yval)
                    plt.text(bar.get_x() + bar.get_width()/2.0, yval_float + 0.01, f'{yval_float:.3f}', ha='center', va='bottom')
                except (ValueError, TypeError):
                    plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.01, 'N/A', ha='center', va='bottom')


            output_filename = f"infographics/{safe_sheet_name}_r2_comparison.png"
            plt.savefig(output_filename)
            print(f"Saved chart: {output_filename}")
            plt.close()
        else:
            print(f"Skipping sheet '{sheet_name}' as there is no data to plot after NaN removal.")
            plt.close() # Close the figure if no data to plot
    else:
        print(f"Skipping sheet '{sheet_name}' due to missing 'Folder' or 'Test R2' columns.")
        plt.close() # Close the figure if columns are missing

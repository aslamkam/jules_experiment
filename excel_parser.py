import pandas as pd

def main():
    excel_filename = "Model_Results_Tabulation.xlsx"
    output_filename = "excel_summary.txt"

    try:
        excel_file = pd.ExcelFile(excel_filename)
    except FileNotFoundError:
        with open(output_filename, "w") as f:
            f.write(f"Error: Excel file '{excel_filename}' not found.\n")
        print(f"Error: Excel file '{excel_filename}' not found.")
        return

    with open(output_filename, "w") as outfile:
        outfile.write("Sheet names: " + ", ".join(excel_file.sheet_names) + "\n\n")
        print("Sheet names:", excel_file.sheet_names)

        for sheet_name in excel_file.sheet_names:
            try:
                df = excel_file.parse(sheet_name)
                outfile.write(f"--- {sheet_name} ---\n")
                outfile.write(df.head().to_string())
                outfile.write("\n\n")
                print(f"--- {sheet_name} ---")
                print(df.head())
            except Exception as e:
                error_message = f"Error processing sheet '{sheet_name}': {e}\n"
                outfile.write(error_message)
                print(error_message)

if __name__ == "__main__":
    main()

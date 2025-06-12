import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from matplotlib.ticker import MaxNLocator

def processor(ROOT):

    raw_data_path = os.path.join("data", "raw", "munich_accidents.csv")
    processed_data_path = os.path.join("data", "processed")
    report_path = os.path.join(ROOT, "reports")

    # Create directories if they don't exist
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    if not os.path.exists(report_path):
        os.makedirs(report_path)

    def load_and_clean_data(ROOT, raw_data_path, processed_data_path):

        df = pd.read_csv(os.path.join(ROOT, raw_data_path))

        # Keep only the 5 relevant columns
        df = df[['MONATSZAHL', 'AUSPRAEGUNG', 'JAHR', 'MONAT', 'WERT']]

        # Keep only rows with numeric MONAT
        df = df[df['MONAT'].astype(str).str.isnumeric()]
        df['MONAT'] = df['MONAT'].astype(int)

        # Extract month (last 2 digits of MONAT)
        df['MONAT'] = df['MONAT'] % 100

        # Check and confirm that the rows for MONTH < 12
        df = df[df['MONAT'] <= 12]

        # Rename columns to English
        df.rename(columns={
        'MONATSZAHL': 'Category',
        'AUSPRAEGUNG': 'Accident_type',
        'JAHR': 'Year',
        'MONAT': 'Month',
        'WERT': 'Value'
        }, inplace=True)

        # Sort and reset index
        df = df.sort_values(by=['Year', 'Month']).reset_index(drop=True)

        gt = df[
                (df['Category'] == 'Alkoholunfälle') &
                (df['Accident_type'] == 'insgesamt') &
                (df['Year'] == 2021) &
                (df['Month'] == 1)
            ].copy()

        if not gt.empty:
            value = gt['Value'].values[0]

            ground_truth_data = {
                                "Category": "Alkoholunfälle",
                                "Accident_type": "insgesamt",
                                "Year": 2021,
                                "Month": 1,
                                "Value": int(value)
                                }

            with open(os.path.join(report_path, "ground_truth_2021_01.json"), "w", encoding="utf-8") as f:
                json.dump(ground_truth_data, f, ensure_ascii=False, indent=2)
            
            print(f"[✔] Ground truth saved to: {report_path}")
        else:
            print("[!] No ground truth found for Jan 2021 Alkoholunfälle")
        
        df.to_csv(os.path.join(ROOT, processed_data_path,  "test_cleaned_data.csv"), index=False)
        print(f"[✔] Test cleaned data saved to: {os.path.join(ROOT, processed_data_path)}")

        # Keep data only till 2020
        df = df[df['Year'] <= 2020]

        df.to_csv(os.path.join(ROOT, processed_data_path,  "cleaned_data.csv"), index=False)
        print(f"[✔] Cleaned data saved to: {os.path.join(ROOT, processed_data_path)}")

    def generate_visualization(ROOT, processed_data_path, report_path):

        df = pd.read_csv(os.path.join(ROOT, processed_data_path,  "cleaned_data.csv"))

        # Group and sum accident values by year and category
        df_viz = df.groupby(['Year', 'Category'])['Value'].sum().unstack()

        # Plot
        df_viz.plot(figsize=(12, 6), marker='o')
        plt.title('Yearly Traffic Accidents by Category (Munich)')
        plt.ylabel('Total Accidents')
        plt.xlabel('Year')
        plt.grid(True)
        plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.savefig(os.path.join(report_path, "historical_accidents_per_category.png"), 
                                dpi = 200)
        plt.clf()   
        plt.close()
        print(f"[✔] Visualization saved to: {report_path}")

    load_and_clean_data(ROOT, raw_data_path, processed_data_path)
    generate_visualization(ROOT, processed_data_path, report_path)
from pathlib import Path

INPUT_FILE = Path("data/raw/Churn_Modelling.csv.xls")
OUTPUT_FILE = Path("data/raw/Churn_Modelling.csv")

print("Fixing Dataset 1...")

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line in infile:
        line = line.strip()

        # Remove surrounding quotes
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]

        outfile.write(line + "\n")

print("\nDataset repaired successfully.")
print(f"Saved to: {OUTPUT_FILE}")
# Auto-generated equivalent of Ashaar_Colab_Demo.ipynb

import os
import sys
from pathlib import Path

from Ashaar.bait_analysis import BaitAnalysis
from Ashaar.utils import get_output_df

def main() -> None:
    # Reduce TensorFlow logging noise
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir / "Ashaar"

    # Ensure we run from the Ashaar project root so relative paths work
    os.chdir(project_root)

    # Make the project importable as a package (Ashaar, poetry_diacritizer, etc.)
    sys.path.insert(0, str(project_root))


    # Example bait from the notebook
    baits = [
        "قَرِيبٌ إِلَى رَبٍّ رَؤوفٍ بِفَضْلِهِ # رَفِيعٌ إِلَى مَجْدٍ سَنَاءٍ بِمَجْدِهِ",
    ]

    analysis = BaitAnalysis(abs_path=".")
    output = analysis.analyze(baits, override_tashkeel=True)

    # Convert analysis to a DataFrame like in the notebook
    df = get_output_df(output)
    print(df)

    # Save results to CSV and TXT in the script directory
    csv_path = base_dir / "bait_analysis_output.csv"
    txt_path = base_dir / "bait_analysis_output.txt"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_csv(txt_path, index=False, sep="\t", encoding="utf-8-sig")
    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved TXT to: {txt_path}")


if __name__ == "__main__":
    main()

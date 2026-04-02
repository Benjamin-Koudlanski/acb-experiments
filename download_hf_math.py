import os
import json
import subprocess
import sys
from pathlib import Path

# 1. Installation automatique de la librairie
try:
    from datasets import load_dataset
except ImportError:
    print("📦 Installation de la librairie 'datasets'...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

# 2. Le dossier cible
base_dir = Path("benchmarks/data/MATH")

print("📥 Téléchargement du dataset...")
ds = load_dataset("qwedsacf/competition_math")

# On regarde ce qu'il y a vraiment dans le dataset (train, test, validation ?)
available_splits = ds.keys()
print(f"🔍 Sections trouvées : {list(available_splits)}")

# 3. Extraction
for split in available_splits:
    # Ton code P2 attend un dossier "test", donc si on trouve "train" ou "validation",
    # on va quand même mettre les fichiers dans un dossier nommé "test" pour le débloquer.
    folder_name = "test"
    print(f"📂 Conversion de la section '{split}' vers le dossier '{folder_name}'...")

    for i, row in enumerate(ds[split]):
        # Nettoyage du nom de catégorie
        category = str(row.get("type", "Other")).replace(" ", "_").replace("/", "_")
        cat_dir = base_dir / folder_name / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        file_path = cat_dir / f"{i}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({
                "problem": row["problem"],
                "level": row["level"],
                "type": row["type"],
                "solution": row["solution"]
            }, f, ensure_ascii=False, indent=2)

print(f"✅ Terminé ! Le dataset est prêt dans {base_dir}/test")
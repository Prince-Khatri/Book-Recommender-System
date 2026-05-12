from contextlib import redirect_stdout
import importlib

files = [
    "item_based_cf",
    "user_based_cf",
    "hybrid"
]

with open("report.txt", "w") as report:

    for file in files:

        report.write(f"\n{'='*20} {file} {'='*20}\n\n")

        # Capture all print statements
        with redirect_stdout(report):

            importlib.import_module(file)

        report.write("\n\n")
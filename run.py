# run.py
import os

def run_script(script_name):
    """ Function to run a Python script using the os.system command """
    status = os.system(f"python {script_name}")
    if status != 0:
        print(f"Error running {script_name}")

def main():
    # List of Python scripts to run
    scripts = [
        "sentiment_classification.py",
        "text_analysis.py",
        "mia.py"
    ]

    # Loop through the scripts and run them
    for script in scripts:
        run_script(script)

if __name__ == "__main__":
    main()


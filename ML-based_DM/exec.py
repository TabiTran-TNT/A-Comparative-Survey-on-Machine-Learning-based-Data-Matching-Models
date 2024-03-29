import time
import subprocess

def run_and_measure_time(script_path):
    start_time = time.time()

    # Run the other Python script using subprocess
    subprocess.run(['python', script_path])

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Execution time: {elapsed_time:.2f} seconds")
choice = 0
script_paths = ["./NaiveB.py", "./RandomForest.py", "./DecisionT.py", "./LogisticR.py", "./SVM.py", "./XGBoost.py"]
if (choice == 1):
    script_paths = ["./NaiveB.py", "./RandomForest.py", "./DecisionT.py", "./LogisticR.py", "./XGBoost.py"]
elif (choice == 2):
    script_paths = ["./SVM.py"]

if __name__ == "__main__":
    # Replace 'path_to_your_script.py' with the actual path to your Python script
    for script in script_paths:   
        print("----------------------------") 
        print(script)
        run_and_measure_time(script)

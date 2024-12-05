import argparse
from src.codes.decision_tree.decision_tree_model import train_decision_tree
from src.codes.svm.svm_model import train_svm
from src.codes.random_forest.random_forest_model import train_random_forest

def main(model_name):
    if model_name == "decision_tree":
        train_decision_tree()
    elif model_name == "svm":
        train_svm()
    elif model_name == "random_forest":
        train_random_forest()
    else:
        print(f"Model '{model_name}' not recognized.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a specified model.")
    parser.add_argument("--model", type=str, required=True, help="Specify model type: decision_tree, svm, or random_forest")
    args = parser.parse_args()
    main(args.model)

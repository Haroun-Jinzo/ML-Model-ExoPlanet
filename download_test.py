import pandas as pd

def generate_test_sample():

    df = pd.read_csv("ModelTesting.csv")
    test_sample = df.drop('Label', axis=1).iloc[0:1]
    test_sample.to_csv("test_sample.csv", index=False, header=False)
    print("Test sample saved to test_sample.csv")

if __name__ == "__main__":
    generate_test_sample()
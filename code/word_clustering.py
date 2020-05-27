import pandas as pd


def main():
    df = pd.read_csv('vectors.txt', sep=' ')
    print(df.head())
    print(df.describe())

if __name__ == '__main__':
    main()

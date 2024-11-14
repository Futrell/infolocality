import sys
import csv

import pandas as pd

INDEX_COLS = ['Words']
VALUE_COLS = "AROU VAL DOM CNC IMAG FAM SIZE GEND".split()

def main(filename, index_columns=INDEX_COLS, value_columns=VALUE_COLS):
    df = pd.read_csv(filename)
    y = df[index_columns]
    for column in value_columns:
        m = df[column].mean()
        y[column] = (df[column] > m).map(int)
    y.to_csv(sys.stdout, index=False)

if __name__ == '__main__':
    main(*sys.argv[1:])

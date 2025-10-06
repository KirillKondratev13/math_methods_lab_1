#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import pandas as pd

# Mapping for observation groups
group_mapping = {
    '0': 'хрАГ',
    '1': 'здоровые',
    '2': 'ГАГ',
    '3': 'ум.ПЭ',
    '4': 'тяж.ПЭ'
}

def main():
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    y_train = data['y_train']
    y_test = data['y_test']
    y_all = pd.concat([pd.Series(y_train), pd.Series(y_test)], ignore_index=True)

    print(f"Total number of patients: {len(y_all)}")
    print("Patient IDs: Not available (removed during preprocessing)")
    print("Count of patients in each observation group:")
    counts = y_all.value_counts().sort_index()
    for group_num, count in counts.items():
        print(f"{group_mapping[group_num]}: {count}")

if __name__ == "__main__":
    main()
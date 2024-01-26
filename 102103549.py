import sys
import pandas as pd
import numpy as np

def normalize(matrix):
    return matrix / np.linalg.norm(matrix, axis=0)

def weighted_normalized_matrix(normalized_matrix, weights):
    return normalized_matrix * weights

def ideal_best_worst(matrix, impacts):
    ideal_best = matrix.max() if impacts == '+' else matrix.min()
    ideal_worst = matrix.min() if impacts == '+' else matrix.max()
    return ideal_best, ideal_worst

def euclidean_distance(matrix, ideal_value):
    return np.sqrt(np.sum((matrix - ideal_value) ** 2, axis=1))

def topsis(input_file, weights, impacts, output_file):
    try:
        df = pd.read_csv(input_file, index_col=0, encoding='utf-8')
    except FileNotFoundError:
        print("Error: File not found.")
        return
    except UnicodeDecodeError:
        print("Error: Unable to decode the file. Check the file encoding.")
        return

    if df.shape[1] < 3:
        print("Error: Input file must contain three or more columns.")
        return

    try:
        df = df.apply(pd.to_numeric, errors='coerce')
    except ValueError:
        print("Error: Non-numeric values found in the input file.")
        return

    num_columns = df.shape[1] - 1
    num_weights = len(weights.split(','))
    num_impacts = len(impacts.split(','))

    if num_weights != num_columns or num_impacts != num_columns:
        print("Error: Number of weights, impacts, and columns must be the same.")
        return

    # if any(char not in ['+', '-'] for char in impacts):
    #  print("Error: Impacts must be either +ve or -ve.")
    #  return


    weights_array = np.array(list(map(int, weights.split(','))))
    normalized_data = normalize(df.values[:, 1:])
    weighted_data = weighted_normalized_matrix(normalized_data, weights_array)

    ideal_best_val, ideal_worst_val = ideal_best_worst(weighted_data, impacts)
    
    distance_best = euclidean_distance(weighted_data, ideal_best_val)
    distance_worst = euclidean_distance(weighted_data, ideal_worst_val)

    score = distance_worst / (distance_best + distance_worst)
    rank = np.argsort(score) + 1

    result_df = pd.DataFrame({'Object': df.index, 'Topsis Score': score, 'Rank': rank})
    result_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")
    else:
        input_file = sys.argv[1]
        weights = sys.argv[2]
        impacts = sys.argv[3]
        output_file = sys.argv[4]
        topsis(input_file, weights, impacts, output_file)

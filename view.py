import os
import matplotlib.pyplot as plt
import math
import numpy as np


def read_data_from_file(file_path):
    """Reads data from a text file."""
    with open(file_path, "r") as file:
        data = file.readlines()
    return [float(line.strip()) for line in data]


def plot_all_data(files_data, indices):
    """Plots data from multiple files on the same graph and marks the provided indices."""
    for file_name, data in files_data.items():
        time = [i * 0.25 for i in range(len(data))]  # Time with a step of 0.25 seconds
        plt.plot(
            time, data, marker="o", linestyle="-", label=file_name
        )  # Use file_name in the legend

        # Plot vertical lines for provided indices
        if file_name in indices:
            for index, value in indices[file_name]:
                plt.axvline(
                    x=index * 0.25,
                    color="r" if value == "x_max" else "b",
                    linestyle="--",
                    label=f"{file_name}: {value} at t={index * 0.25}s",
                )

    plt.title("Plot of data from multiple files")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def find_last_index_less_than(arr, x):
    arr = np.array(arr)
    indices = np.where(arr < x)[0]  # находим индексы, где значение < x
    if len(indices) == 0:
        return -1  # если таких элементов нет
    return indices[-1]  # возвращаем последний индекс


def find_last_index(arr, x):
    arr = np.array(arr)  # преобразуем вход в numpy-массив, если это список
    indices = np.where(arr == x)[0]  # находим все индексы, где значение равно x
    if len(indices) == 0:
        return -1  # если x не найден, возвращаем -1
    return indices[-1]  # возвращаем последний индекс


def get_rounding_step(x):
    """Determines the rounding step based on the number of decimal places."""
    x_str = str(x)
    if "." in x_str:
        decimals = len(x_str.split(".")[-1])
        return 10**-decimals
    return 1  # If x is an integer


def find_last_index_less(arr, x):
    arr = np.array(arr, dtype=float)
    step = get_rounding_step(x)

    # Quantize the array
    quantized_arr = np.floor(arr / step) * step
    # Find indices where values are less than x
    indices = np.where(quantized_arr < x)[0]
    # Compute the difference between consecutive indices
    diff = np.diff(indices)
    # Find the last index in diff where the value is not equal to 1
    last_diff_index = -1
    for i in range(len(diff) - 1, -1, -1):
        if diff[i] != 1:
            last_diff_index = i
            break

    # If found, add it to the first index from indicesssssssss
    if len(indices) > 0 and last_diff_index != -1:
        adjusted_index = indices[last_diff_index + 1]
        return adjusted_index

    # If nothing is found, return -1
    return -1


if __name__ == "__main__":
    current_directory = os.getcwd()  # Current directory
    txt_files = [f for f in os.listdir(current_directory) if f.endswith(".txt")]
    x_max = 3.3  # Value to find and mark on the graph
    x_min = 0.9  # Value to find and mark on the graph

    if not txt_files:
        print("No .txt files found in the current directory.")
    else:
        files_data = {}
        indices = {}

        for file_name in txt_files:
            try:
                print(f"Reading data from file: {file_name}")
                data = read_data_from_file(file_name)
                files_data[file_name] = data

                # Find indices for x_max and x_min
                last_index_max = find_last_index(data, x_max)
                print(f"Last index with value {x_max}: {last_index_max}")
                last_index_min = find_last_index_less(data, x_min)
                print(f"Last index with value {x_min}: {last_index_min}")

                # Store indices with labels
                indices[file_name] = []
                if last_index_max != -1:
                    indices[file_name].append((last_index_max, "x_max"))
                if last_index_min != -1:
                    indices[file_name].append((last_index_min, "x_min"))
                print(f"Indices for {indices}")

            except FileNotFoundError:
                print(f"File {file_name} not found.")
            except ValueError:
                print(f"Error: File {file_name} contains invalid data.")

        if files_data:
            plot_all_data(files_data, indices)

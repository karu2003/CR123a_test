import os
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
import matplotlib.patches as mpatches


def read_data_from_file(file_path):
    """Reads data from a text file."""
    with open(file_path, "r") as file:
        data = file.readlines()
    return [float(line.strip()) for line in data]


def calculate_energy_from_interpolation(time, voltage, current):
    """
    Calculates the discharge energy of the battery using interpolated data.

    Parameters:
        time (list): Time points in hours.
        voltage (list): Voltage values corresponding to the time points.
        current (float): Discharge current in amperes.

    Returns:
        float: Total discharge energy in joules.
    """
    time_seconds = [t * 3600 for t in time]  # Convert time from hours to seconds
    voltage_integral = simps(voltage, time_seconds)
    return current * voltage_integral  # Joules


def plot_all_data(files_data, indices, energies, discharge_currents):
    """
    Plots data from multiple files on the same graph and annotates the energy values.

    Parameters:
        files_data (dict): Dictionary containing data for each file.
        indices (dict): Dictionary containing indices for annotations.
        energies (dict): Dictionary containing energy values in Wh for each file.
        discharge_currents (dict): Dictionary containing discharge currents for each file.
    """
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    for file_name, data in files_data.items():
        time = [i * 0.25 / 3600 for i in range(len(data))]  # Time in hours

        # Interpolation
        interp_func = interp1d(time, data, kind="cubic", fill_value="extrapolate")
        interpolated_time = np.linspace(min(time), max(time), 500)
        interpolated_data = interp_func(interpolated_time)

        # Energy
        discharge_current = discharge_currents[file_name]
        energy_wh_interpolated = calculate_energy_from_interpolation(
            interpolated_time, interpolated_data, discharge_current
        )

        energy_wh = energies[file_name]
        
        # Plotting the original data
        ax.plot(time, data, ".", label=f"{file_name} (Original Data)", alpha=0.7)

        # Plotting the graph
        (line,) = ax.plot(interpolated_time, interpolated_data, "-", label=file_name)

        # Coordinates of the right end of the line (for annotation)
        x_text = interpolated_time[-1]
        y_text = interpolated_data[-1]

        # Shift the text slightly to the right and up
        text_offset = 0.01  # along X (hours)
        vertical_offset = 0.05  # along Y (volts)

        label = f"{discharge_current}A load\n{energy_wh:.3f} Wh"
        ax.annotate(
            label,
            xy=(x_text, y_text),
            xytext=(x_text + text_offset, y_text + vertical_offset),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color=line.get_color(), lw=1.5),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=line.get_color(), lw=1),
            color=line.get_color(),
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    plt.title("Battery CR123A Discharge Graph")
    plt.xlabel("Time (h)")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("battery_discharge_graph.png", dpi=300)
    plt.show()


def find_last_index_less_than(arr, x):
    """
    Finds the last index in the array where the value is less than x.

    Parameters:
        arr (list or np.array): Array of values.
        x (float): Threshold value.

    Returns:
        int: Last index where the value is less than x, or -1 if not found.
    """
    arr = np.array(arr)
    indices = np.where(arr < x)[0]  # Find indices where the value is < x
    if len(indices) == 0:
        return -1  # If no such elements exist
    return indices[-1]  # Return the last index


def find_last_index(arr, x):
    """
    Finds the last index in the array where the value equals x.

    Parameters:
        arr (list or np.array): Array of values.
        x (float): Target value.

    Returns:
        int: Last index where the value equals x, or -1 if not found.
    """
    arr = np.array(arr)  # Convert input to a numpy array if it's a list
    indices = np.where(arr == x)[0]  # Find all indices where the value equals x
    if len(indices) == 0:
        return -1  # If x is not found, return -1
    return indices[-1]  # Return the last index


def get_rounding_step(x):
    """
    Determines the rounding step based on the number of decimal places.

    Parameters:
        x (float): Input value.

    Returns:
        float: Rounding step.
    """
    x_str = str(x)
    if "." in x_str:
        decimals = len(x_str.split(".")[-1])
        return 10**-decimals
    return 1  # If x is an integer


def find_last_index_less(arr, x):
    """
    Finds the last index in the array where the quantized value is less than x.

    Parameters:
        arr (list or np.array): Array of values.
        x (float): Threshold value.

    Returns:
        int: Last index where the quantized value is less than x, or -1 if not found.
    """
    arr = np.array(arr, dtype=float)
    step = get_rounding_step(x)

    # Quantize the array
    quantized_arr = np.floor(arr / step) * step
    # Find indices where values are less than x
    indices = np.where(quantized_arr < x)[0]
    # Compute the difference between consecutive indices
    diff = np.diff(indices)
    # Find the last index in diff where the value is not equal to 1
    last_diff_index = -1  # Default to -1
    for i in range(len(diff) - 1, -1, -1):
        if diff[i] != 1:
            last_diff_index = i
            break

    # If found, add it to the first index from indices
    if len(indices) > 0:
        if last_diff_index != -1:
            adjusted_index = indices[last_diff_index + 1]
        else:
            adjusted_index = indices[0]  # Use the first element of indices
        return adjusted_index

    # If nothing is found, return -1
    return -1


def calculate_discharge_energy(data, discharge_current, time_step=0.25):
    """
    Calculates the discharge energy of the battery.

    Parameters:
        data (list): Voltage values over time.
        discharge_current (float): Discharge current in amperes.
        time_step (float): Time step between measurements in seconds.

    Returns:
        float: Total discharge energy in joules.
    """
    energy = 0.0
    for voltage in data:
        energy += voltage * float(discharge_current) * time_step
    return energy


def calculate_discharge_energy_integral(data, discharge_current, time_step=0.25):
    """
    Calculates the discharge energy of the battery using numerical integration.

    Parameters:
        data (list): Voltage values over time.
        discharge_current (float): Discharge current in amperes.
        time_step (float): Time step between measurements in seconds.

    Returns:
        float: Total discharge energy in joules.
    """
    # Time array
    time = np.arange(0, len(data) * time_step, time_step)
    # Integrate voltage * current
    energy = simps([v * float(discharge_current) for v in data], time)
    return energy


def calculate_energy_from_interpolation(interpolated_time, interpolated_data, discharge_current):
    """
    Calculates the discharge energy of the battery using interpolated data.

    Parameters:
        interpolated_time (list): Interpolated time points.
        interpolated_data (list): Interpolated voltage values.
        discharge_current (float): Discharge current in amperes.

    Returns:
        float: Total discharge energy in joules.
    """
    # Integrate voltage * current over time
    energy = simps(
        [v * float(discharge_current) for v in interpolated_data], interpolated_time
    )
    return energy


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
        energies = {}
        discharge_currents = {}

        for file_name in txt_files:
            try:
                print(f"Reading data from file: {file_name}")
                data = read_data_from_file(file_name)
                discharge_current = file_name[0]
                discharge_currents[file_name] = discharge_current
                print(f"Discharge current: {discharge_current}")

                # Find indices for x_max and x_min
                last_index_max = find_last_index(data, x_max)
                last_index_min = find_last_index_less(data, x_min)

                # Store indices with labels
                indices[file_name] = []
                if last_index_max != -1:
                    indices[file_name].append((last_index_max, "x_max"))
                if last_index_min != -1:
                    indices[file_name].append((last_index_min, "x_min"))
                valid_data = data[last_index_max:last_index_min]
                files_data[file_name] = valid_data

                # Calculate discharge energy in joules
                energy_joules = calculate_discharge_energy(
                    valid_data, discharge_current
                )

                # Calculate discharge energy using integration
                energy_joules = calculate_discharge_energy_integral(
                    valid_data, discharge_current
                )
                # Convert energy to watt-hours
                energy_wh = energy_joules / 3600
                energies[file_name] = energy_wh
                print(
                    f"Discharge energy for {file_name}: {energy_joules:.2f} J ({energy_wh:.4f} Wh)"
                )

            except FileNotFoundError:
                print(f"File {file_name} not found.")
            except ValueError:
                print(f"Error: File {file_name} contains invalid data.")

        if files_data:
            plot_all_data(files_data, indices, energies, discharge_currents)

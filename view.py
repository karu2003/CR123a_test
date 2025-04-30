import os
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
import matplotlib.patches as mpatches
import csv
from matplotlib.ticker import FuncFormatter


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

def calculate_energy(voltage_list, time_list, current_a):
    energy_wh = 0.0
    for i in range(len(voltage_list) - 1):
        u = voltage_list[i]
        delta_t = time_list[i+1] - time_list[i]
        energy_wh += u * current_a * delta_t  # Вт·ч
    return energy_wh


def plot_all_data(files_data, indices, energies, discharge_currents, org_data_dict):
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
    log_ticks = []

    # List to store occupied annotation coordinates (x, y)
    annotated_points = []

    # Обработка данных из txt файлов
    for file_name, data in files_data.items():
        time = [i * 0.25 / 3600 for i in range(len(data))]  # Time in hours
        log_ticks.append(time[-1])

        # Interpolation
        interp_func = interp1d(time, data, kind="cubic", fill_value="extrapolate")
        interpolated_time = np.linspace(min(time), max(time), 500)
        interpolated_data = interp_func(interpolated_time)

        # Energy
        discharge_current = extract_current_from_filename(file_name)
        energy_wh_interpolated = calculate_energy_from_interpolation(
            interpolated_time, interpolated_data, discharge_current
        )

        energy_wh = energies[file_name]

        # Plotting the original data and graph
        ax.plot(time, data, ".", label=f"{file_name} (Original Data)", alpha=0.7)
        (line,) = ax.plot(interpolated_time, interpolated_data, "-", label=file_name)

        # Координаты конца линии для аннотации
        x_text = interpolated_time[-1]
        y_text = interpolated_data[-1]

        # Улучшенное размещение аннотаций
        vertical_offset = 20.08  # Увеличено
        horizontal_offset = 0.03  # Увеличено
        min_dist = 0.05  # Увеличено
        orig_y = y_text
        orig_x = x_text
        max_attempts = 100  # Увеличено

        attempts = 0
        direction = 1
        while any(
            abs(y_text - y) < min_dist and abs(x_text - x) < 0.1  # Увеличено
            for x, y in annotated_points
        ) and attempts < max_attempts:
            # Чередуем вверх/вниз и вправо/влево
            if attempts % 2 == 0:
                y_text += direction * vertical_offset
            else:
                x_text += direction * horizontal_offset

            direction *= -1
            # Увеличиваем смещения при большом количестве попыток
            if attempts % 10 == 0 and attempts > 0:
                vertical_offset *= 1.5
                horizontal_offset *= 1.5

            attempts += 1

        annotated_points.append((x_text, y_text))
        
        label = f"{discharge_current} A\n{energy_wh:.3f} Wh"
        ax.annotate(
            label,
            xy=(orig_x, orig_y),
            xytext=(x_text, y_text),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color=line.get_color(), lw=1.5),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=line.get_color(), lw=1),
            color=line.get_color(),
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    # Обработка данных из org_data_dict - ИСПОЛЬЗУЕМ ТОТ ЖЕ АЛГОРИТМ для избежания наложений
    for label, data in org_data_dict.items():
        current_a = float(label.split()[0]) / 1000   
        energy = calculate_energy(data["voltage"], data["time"], current_a)
        text_label = f"{label}\n{energy:.3f} Wh"
        (line,) = ax.plot(data["time"], data["voltage"], label=label)
        log_ticks.append(data["time"][-1])

        # Координаты для аннотации
        x_text = data["time"][-1]
        y_text = data["voltage"][-1]
        orig_x = x_text
        orig_y = y_text

        # Используем тот же алгоритм предотвращения наложений
        vertical_offset = 0.08
        horizontal_offset = 0.03
        min_dist = 0.05
        max_attempts = 100
        attempts = 0
        direction = 1

        while any(
            abs(y_text - y) < min_dist and abs(x_text - x) < 0.1
            for x, y in annotated_points
        ) and attempts < max_attempts:
            # Чередуем вверх/вниз и вправо/влево
            if attempts % 2 == 0:
                y_text += direction * vertical_offset
            else:
                x_text += direction * horizontal_offset

            direction *= -1
            # Увеличиваем смещения при большом количестве попыток
            if attempts % 10 == 0 and attempts > 0:
                vertical_offset *= 1.5
                horizontal_offset *= 1.5

            attempts += 1

        annotated_points.append((x_text, y_text))

        # Добавляем аннотацию 
        ax.annotate(
            text_label,
            xy=(orig_x, orig_y),
            xytext=(x_text, y_text),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color=line.get_color(), lw=1.5),
            fontsize=10,
            color=line.get_color(),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=line.get_color(), lw=1),
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    ax.set_xscale("log")
 
    # Set custom ticks for the X-axis
    log_ticks.sort()
    ax.set_xticks(log_ticks)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))

    # Set X-axis limits to match the data range
    # min_time = min(min(data["time"]) for data in org_data_dict.values())
    min_time = 0.04  # Ensure left limit is >= 1e-3
    max_time = max(max(data["time"]) for data in org_data_dict.values())
    ax.set_xlim(
        left=max(min_time, 1e-3), right=max_time
    )  # Ensure left limit is >= 1e-3

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


def calculate_energy_from_interpolation(
    interpolated_time, interpolated_data, discharge_current
):
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


def extract_current_from_filename(file_name):
    """
    Extracts current value from filename.
    Examples:
    - 1A.txt returns 1.0
    - 1_5A.txt returns 1.5
    - 0_5A.txt returns 0.5
    
    Parameters:
        file_name (str): Name of the file
        
    Returns:
        float: Current value in amperes
    """
    # Remove file extension
    name_without_ext = file_name.split('.')[0]
    # Remove 'A' suffix
    name_without_a = name_without_ext.replace('A', '')
    # Replace underscore with decimal point
    current_str = name_without_a.replace('_', '.')
    # Convert to float
    try:
        return float(current_str)
    except ValueError:
        print(f"Could not extract current from filename: {file_name}")
        return 0.0


if __name__ == "__main__":
    current_directory = os.getcwd()  # Current directory
    txt_files = [f for f in os.listdir(current_directory) if f.endswith(".txt")]
    x_max = 3.3  # Value to find and mark on the graph
    x_min = 0.9  # Value to find and mark on the graph

    # Path to the CSV file
    filename = "CR123A.csv"

    # Separate lists for each curve
    t_300, v_300 = [], []
    t_100, v_100 = [], []
    t_500, v_500 = [], []

    # Reading and filtering
    with open(filename, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                t = float(row[0].replace(",", "."))
                v300 = float(row[1].replace(",", "."))
                v100 = float(row[2].replace(",", "."))
                v500 = float(row[3].replace(",", "."))

                # Add to each graph independently if the value is valid
                if v300 >= 0:
                    t_300.append(t)
                    v_300.append(v300)
                if v100 >= 0:
                    t_100.append(t)
                    v_100.append(v100)
                if v500 >= 0:
                    t_500.append(t)
                    v_500.append(v500)

            except (ValueError, IndexError):
                continue

    org_data_dict = {
        "300 mA": {"time": t_300, "voltage": v_300},
        "100 mA": {"time": t_100, "voltage": v_100},
        "500 mA": {"time": t_500, "voltage": v_500},
    }

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
                discharge_current = extract_current_from_filename(file_name)
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
            plot_all_data(
                files_data, indices, energies, discharge_currents, org_data_dict
            )

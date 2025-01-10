import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from PyLTSpice import RawRead


class MemCrossbarSim:

    def __init__(self, netlist_file, input_voltages_file, data_file_raw, data_file_csv, bias_value=0.1,
                 learning_rate=0.01, epochs=20, batch_size=10):
        self.netlist_file = netlist_file
        self.input_voltages_file = input_voltages_file
        self.data_file_raw = data_file_raw
        self.data_file_csv = data_file_csv
        self.bias_value = bias_value
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.losses = []

    def initialize_netlist(self, input_voltages_df):
        """
        Initialize the netlist file for LTSpice simulation.

        Args:
        - input_voltages_df: Initial pattern Dataframe for updating input voltages.

        This method updates input voltages based on the initial pattern,
        initializes memristors with high resistance values, and updates the
        netlist file accordingly.

        """

        input_voltages = input_voltages_df.sample(1).values.flatten()
        # Update input voltages based on initial pattern:
        self.update_input_voltages(input_voltages)

        # Initializing memristors weights to high value ~ 1e6:
        num_memristors = self.count_memristors()
        memristor_values = np.random.uniform(0, 1, num_memristors) * 1e6 # Value scaled for significant change

        with open(self.netlist_file, 'r', encoding='utf-8') as file:
            netlist_content = file.readlines()

        updated_netlist_content = []
        memristor_index = 0

        # Process each line in the netlist file
        for line in netlist_content:
            clean_line = line.replace('\x00', '').strip()

            # Update memristor values for lines containing 'Xmem' and 'memristor_simple'
            if clean_line.startswith('Xmem') and 'memristor_simple' in clean_line:
                parts = clean_line.split()

                # Check if there is a bias specified in the line
                bias_match = re.search(r'bias(-?\d+\.\d+)', clean_line)
                if bias_match:
                    current_bias = bias_match.group(0)
                    updated_line = re.sub(r'bias(-?\d+\.\d+)', f'bias{self.bias_value}', clean_line)
                    updated_netlist_content.append(updated_line + '\n')
                else:
                    # Update resistance values with initialized memristor values
                    for index, part in enumerate(parts):
                        if part.startswith('R='):
                            parts[index] = f'R={memristor_values[memristor_index]}'
                            memristor_index += 1
                    updated_line = ' '.join(parts) + '\n'
                    updated_netlist_content.append(updated_line)
            else:
                # Preserve non-memristor lines as is
                updated_netlist_content.append(line)

        with open(self.netlist_file, 'w', encoding='utf-8') as file:
            file.writelines(updated_netlist_content)

    def update_input_voltages(self, input_voltages):
        """
        Update input voltages in the netlist file based on the provided dictionary.

        Args:
        - input_voltages (dict): Dictionary containing input voltages for each label.

        This method reads the netlist file, updates input voltages based on the provided dictionary,
        and writes the updated content back to the netlist file.

        """
        # Read the existing netlist file
        with open(self.netlist_file, 'r', encoding='utf-8') as file:
            netlist_content = file.readlines()

        updated_netlist_content = []

        for line in netlist_content:
            clean_line = line.replace('\x00', '').strip()
            updated = False

            for i in range(len(input_voltages)):
                if f'Vvin{i} in0{i} 0 DC=' in clean_line:
                    updated_netlist_content.append(f'Vvin{i} in0{i} 0 DC={input_voltages[i]}\n')
                    updated = True
                    break

            if not updated:
                updated_netlist_content.append(line)

        with open(self.netlist_file, 'w', encoding='utf-8') as file:
            file.writelines(updated_netlist_content)

    def update_memristors(self, deltas, inputs, layer):
        """
        Update memristor resistances based on deltas and inputs for a specific layer based on the paper:
        https://www.sciencedirect.com/science/article/pii/S0026269216301732

        Args:
        - deltas (numpy.ndarray): Array of delta values to update memristor resistances.
        - inputs (numpy.ndarray): Array of input values corresponding to the deltas.
        - layer (str): Layer identifier ('1' or '2') specifying which memristor layer to update.

        This method calculates updates for memristor resistances based on given deltas and inputs,
        then calls 'mem_net_update' to apply these updates to the LTSpice netlist file.

        """
        # Determine the start and end indices for the memristors in the specified layer
        start_idx = (int(layer) - 1) * 9
        end_idx = start_idx + 9

        updates = []

        # Iterate over the memristors in the specified layer
        for i in range(start_idx, end_idx):
            # Calculate delta and corresponding update for each memristor
            delta = deltas[i - start_idx]
            update = self.learning_rate * delta * inputs[i - start_idx]

            # Add both positive and negative updates to the list
            updates.append(update)
            updates.append(-update)

        # Apply the calculated updates to the memristors in the netlist file
        self.mem_net_update(layer, updates)

    def mem_net_update(self, layer, updates):
        """
        Update memristor resistances in the netlist file based on given updates.

        Args:
        - layer (str): The layer number ('1' or '2') indicating which crossbar to update.
        - updates (list): List of delta values to update memristor resistances.

        This method reads the netlist file, identifies memristor lines corresponding to the given layer,
        calculates new resistance values based on the updates, and writes the updated lines back to the file.

        Error Handling:
        - Handles FileNotFoundError if the netlist file is not found.
        - Handles IOError for any other file access issues.
        - Catches unexpected exceptions and prints an error message.

        Note: This method does not return any values; it directly updates the netlist file.

        """

        try:
            # Define the prefix for the memristor line based on the layer
            if layer == '1':
                mem_prefix = 'Xmem0'
            elif layer == '2':
                mem_prefix = 'Xmem1'
            else:
                print(f"Error: Invalid layer '{layer}'. Layer should be either '1' or '2'.")
                return

            updated_memristors = []

            with open(self.netlist_file, 'r', encoding='utf-8') as file:
                netlist_content = file.readlines()

            for line in netlist_content:
                clean_line = line.replace('\x00', '').strip()

                # Check if the line starts with mem_prefix and contains 'memristor_simple'
                if clean_line.startswith(mem_prefix) and 'memristor_simple' in clean_line:
                    parts = clean_line.split()
                    bias_match = re.search(r'bias(-?\d+\.\d+)', clean_line)

                    # Skip lines with bias
                    if bias_match:
                        updated_memristors.append(line)
                        continue

                    # Extract the index from the line after mem_prefix
                    index_str = parts[0][len(mem_prefix)]
                    index = int(index_str)  # Extract the second digit as integer

                    # Determine if 'n' or 'p' is present in parts
                    found_np = False
                    for part in parts:
                        if 'n' in part or 'p' in part:
                            found_np = True
                            if part.endswith('n'):
                                current_update_index = 2 * index + 1  # Use the odd index (1, 3, 5, ...)
                            elif part.endswith('p'):
                                current_update_index = 2 * index  # Use the even index (0, 2, 4, ...)
                            else:
                                continue  # Skip if neither 'n' nor 'p' is found

                            # Check if current_update_index is within bounds
                            if current_update_index >= len(updates):
                                """
                                print(
                                    f"Warning: update_index {current_update_index} out of range for updates with length {len(updates)}")
                                    """
                                updated_memristors.append(line)
                                break

                            # Apply the update to the resistance value
                            current_resistance = float(parts[-1].split('=')[1])
                            new_resistance = current_resistance + updates[current_update_index]

                            # Modify the resistance value in the line
                            parts[-1] = f'R={new_resistance}'
                            updated_line = ' '.join(parts) + '\n'
                            updated_memristors.append(updated_line)
                            break  # Exit the loop after processing one line

                    if not found_np:
                        updated_memristors.append(line)

                else:
                    updated_memristors.append(line)

            # Write back the modified lines to the netlist file
            with open(self.netlist_file, 'w', encoding='utf-8') as file:
                file.writelines(updated_memristors)

        except FileNotFoundError:
            print(f"Error: File '{self.netlist_file}' not found.")
        except IOError as e:
            print(f"Error: IO error occurred while accessing '{self.netlist_file}': {e}")
        except Exception as e:
            print(f"Error: An unexpected error occurred: {e}")

    def run_ltspice_simulation(self):
        """
        Run LTSpice simulation using the netlist file.

        This method executes LTSpice in batch mode with the specified netlist file,
        capturing the simulation output and printing appropriate messages based on success or failure.

        Returns:
        - str: The stdout output from the LTSpice simulation.

        Raises:
        - subprocess.CalledProcessError: If LTSpice returns a non-zero exit code,
          indicating simulation failure.

        """
        try:
            # Define the command to run LTSpice in batch mode with the netlist file,
            result = subprocess.run(
                [r'~\LTspice.exe', '-b', self.netlist_file],
                capture_output=True, text=True, check=True)

            # If LTSpice simulation is successful, print message and return stdout
            print("LTSpice simulation successful.")
            return result.stdout

        except subprocess.CalledProcessError as e:
            # If LTSpice simulation fails, print error message with return code and stderr
            print(f"LTSpice simulation failed with return code {e.returncode}.")
            print("LTSpice stderr output:", e.stderr)

            # Raise the exception to propagate the error up the call stack
            raise

    def ltspice_raw_to_df(self):
        """
        Convert LTSpice raw data file to a DataFrame and save it as CSV.

        Reads raw data using RawRead, extracts traces, creates a DataFrame,
        and saves it as a CSV file.

        Returns:
        - pd.DataFrame: DataFrame containing LTSpice simulation data.

        """
        # Read raw data from file using RawRead
        L = RawRead(self.data_file_raw)

        # Extract waveforms for each trace and create a dictionary
        data = {trace.name: trace.get_wave(0) for trace in
                (L.get_trace(trace_ref) for trace_ref in L.get_trace_names())}

        # Convert dictionary to DataFrame
        df = pd.DataFrame(data)

        # Save DataFrame as CSV file without index
        df.to_csv(self.data_file_csv, index=False)

        # Return the DataFrame
        return df

    def count_memristors(self):
        """
        Count the number of memristors in the netlist file.

        Opens the netlist file and counts the lines containing 'Xmem' and 'memristor_simple'.

        Returns:
        - int: Number of memristors found in the netlist file.

        """
        with open(self.netlist_file, 'r', encoding='utf-8') as file:
            # Count lines containing 'Xmem' and 'memristor_simple'
            return sum(1 for line in file if 'Xmem' in line and 'memristor_simple' in line)

    def compute_loss(self, sim_voltages, target_voltages):
        """
        Compute the mean squared error loss between simulated and target voltages.

        Args:
        - sim_voltages (np.ndarray): Simulated voltages.
        - target_voltages (np.ndarray): Target voltages.

        Returns:
        - float: Mean squared error loss.

        """
        # Compute mean squared error loss
        return np.mean((sim_voltages - target_voltages) ** 2)

    def adjust_learning_rate(self, epoch, decay_rate=0.8):
        """
        Adjust the learning rate based on the epoch and decay rate. This was an idea to obtain a faster convergence

        Args:
        - epoch (int): Current epoch.
        - decay_rate (float): Rate of decay.

        Returns:
        - float: Adjusted learning rate.

        """
        # Calculate adjusted learning rate using exponential decay
        return self.learning_rate * (decay_rate ** epoch)

    def tanh_derivative(self, x):
        """
        Compute the derivative of the hyperbolic tangent function (tanh).
        """

        # Compute derivative of tanh function
        return 1 - np.tanh(x) ** 2

    def extract_voltages(self, sim_df):
        """
        Extract voltages from simulation DataFrame.

        Args:
        - sim_df (pd.DataFrame): DataFrame containing simulation data.

        Returns:
        - tuple: Tuple containing:
            - dict: Dictionary of bottom voltages:
                - '1+': Array of positive bottom voltages for layer 1.
                - '1-': Array of negative bottom voltages for layer 1.
                - '2+': Array of positive bottom voltages for layer 2.
                - '2-': Array of negative bottom voltages for layer 2.
            - np.ndarray: Array of input 2 cross voltages.

        Raises:
        - ValueError: If required columns are not found in sim_df.

        """
        # Define required column names based on voltage nodes
        required_columns = ['V(er1{}n)'.format(i) for i in range(9)] + ['V(er1{}p)'.format(i) for i in range(9)] + \
                           ['V(er0{}n)'.format(i) for i in range(9)] + ['V(er0{}p)'.format(i) for i in range(9)]

        # Check if all required columns are present in sim_df
        if not all(col in sim_df.columns for col in required_columns):
            raise ValueError("Required columns not found in the simulation data")

        # Extract bottom voltages for each layer and polarity
        bottom_voltages = {
            '1+': sim_df[['V(er0{}p)'.format(i) for i in range(9)]].iloc[-1].astype(float).values,
            '1-': sim_df[['V(er0{}n)'.format(i) for i in range(9)]].iloc[-1].astype(float).values,
            '2+': sim_df[['V(er1{}p)'.format(i) for i in range(9)]].iloc[-1].astype(float).values,
            '2-': sim_df[['V(er1{}n)'.format(i) for i in range(9)]].iloc[-1].astype(float).values
        }

        # Extract cross voltages for input 2
        input_2_cross = sim_df[['V(er0{})'.format(i) for i in range(9)]].iloc[-1].astype(float).values

        # Return extracted voltages
        return bottom_voltages, input_2_cross

    def extract_sigma_layer_1(self):
        """
        Extract conductances difference between negative and positive weights values for layer 1 from the netlist file.
        Useful for the first layer update.

        Returns:
        - np.ndarray: 9x9 matrix of sigma values for layer 1 memristors.

        """
        memristor_sigmas = np.zeros((9, 9))  # Initialize with 9x9 matrix
        crossbar_prefix = 'Xmem'
        memristor_count = 0

        # Read the netlist file
        with open(self.netlist_file, 'r', encoding='utf-8') as file:
            netlist_content = file.readlines()

        # Process each line in the netlist
        for line in netlist_content:
            clean_line = line.replace('\x00', '').strip()

            # Check if the line starts with crossbar_prefix and contains 'memristor_simple'
            if clean_line.startswith(crossbar_prefix) and 'memristor_simple' in clean_line:
                parts = clean_line.split()
                bias_match = re.search(r'bias(-?\d+\.\d+)', clean_line)

                # Skip lines with bias
                if bias_match:
                    continue

                # Extract the resistance value and compute conductance
                memristor_r = float(parts[-1].split('=')[1])
                conductance = 1 / memristor_r

                # Determine the position in the 9x9 matrix
                row = memristor_count // 18  # 18 memristors per row
                col = (memristor_count % 18) // 2  # Each pair (p, n) represents one element in the matrix

                # Handle the conductance values (subtract negative from positive)
                for part in parts:
                    if re.match(r'er\d+n', part):
                        memristor_sigmas[row, col] -= conductance
                    elif re.match(r'er\d+p', part):
                        memristor_sigmas[row, col] += conductance
                    else:
                        continue

                memristor_count += 1

                # Stop processing once we have processed the entire 9x9 matrix
                if memristor_count >= 9 * 18:  # Total of 9 rows * 18 memristors per row
                    break

        return memristor_sigmas

    def run_simulation_and_extract_results(self):
        """
        Run LTSpice simulation, extract simulation results, and process them.

        Returns:
        - np.ndarray: Simulated voltages of output nodes.
        - dict: Voltages at the bottom terminals.
        - np.ndarray: Voltages at input 2 crossbar.

        """
        # Run LTSpice simulation
        self.run_ltspice_simulation()

        # Convert LTSpice raw data to DataFrame
        sim_df = self.ltspice_raw_to_df()

        # Extract simulated voltages of output nodes
        sim_voltages = sim_df[[f'V(out{i:02})' for i in range(9)]].iloc[-1].astype(float).values

        # Extract voltages at bottom terminals and input 2 crossbar
        bottom_voltages, input_2 = self.extract_voltages(sim_df)

        return sim_voltages, bottom_voltages, input_2

    def get_losses(self):
        """
        Get current losses and delta losses from the model.

        Returns:
        - list: Current losses.
        - list: Delta losses.

        """
        # Filter out None values from losses and delta_loss lists
        losses = [loss for loss in self.losses if loss is not None]

        return losses

    def train(self):
        """
        Train the network to reproduce the input pattern using LTSpice simulations and memristor updates.

        """
        input_voltages_df = pd.read_csv(self.input_voltages_file, header=None)
        self.initialize_netlist(input_voltages_df)

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            input_voltages = input_voltages_df.sample(1).values.flatten()
            self.update_input_voltages(input_voltages)

            batch_losses = []

            for _ in range(self.batch_size):
                lr = self.adjust_learning_rate(epoch)
                try:
                    # Run LTSpice simulation and extract results
                    sim_voltages, bottom_voltages, input_2 = self.run_simulation_and_extract_results()
                    target_voltages = np.array(input_voltages, dtype=float)

                    # Calculate target voltages and loss
                    target_loss = self.compute_loss(sim_voltages, target_voltages)
                    batch_losses.append(target_loss)

                    # Calculate dot products and update memristors for layer 2
                    dot_prods = {
                        '1': bottom_voltages['1+'] - bottom_voltages['1-'],
                        '2': bottom_voltages['2+'] - bottom_voltages['2-']
                    }

                    # Deltas for layer 2:
                    delta_2 = (target_voltages - sim_voltages) * self.tanh_derivative(dot_prods['2'])
                    self.update_memristors(delta_2, input_2, '2')

                    # Calculate Sigmas for layer 1 and update memristors
                    Sigmas_1 = self.extract_sigma_layer_1()
                    delta_1 = (Sigmas_1 @ delta_2.T) * self.tanh_derivative(dot_prods['1'])
                    self.update_memristors(delta_1, target_voltages, '1')

                except (FileNotFoundError, subprocess.CalledProcessError, ValueError, RuntimeError, FloatingPointError) as e:
                    print(f"Error: {e}")
                    continue

            if batch_losses:
                # Calculate average loss for the epoch
                avg_loss = np.mean(batch_losses)
                self.losses.append(avg_loss)

    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, marker='o', linestyle='-', color='b', label='Training loss')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_specified_columns(self, columns):
        """
        Plot specified columns from the LTSpice simulation DataFrame.

        Args:
        - columns (list): List of column names to plot.

        This method reads the LTSpice simulation DataFrame from the CSV file,
        extracts and plots the specified columns.

        """
        try:
            # Load the DataFrame from the CSV file
            df = pd.read_csv(self.data_file_csv)

            # Extract the basenames from the columns
            basenames = {}
            for col in columns:
                basename = re.match(r'[^\d]+', col).group(0)
                if basename in basenames:
                    basenames[basename].append(col)
                else:
                    basenames[basename] = [col]

            # Create subplots
            num_plots = len(basenames)
            fig, axs = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))

            if num_plots == 1:
                axs = [axs]

            for ax, (basename, cols) in zip(axs, basenames.items()):
                for col in cols:
                    if col in df.columns:
                        ax.plot(df['time'], df[col], label=col)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Voltage (V)')
                ax.legend()
                ax.set_title(f'Voltage - {basename}')

            plt.tight_layout()
            plt.show()

        except FileNotFoundError as e:
            print(f"Error: {e}")


# ---------------------------------------------------------------------------------------------------------------------

netlist_file = 'double_9x18_crossbar.net'
input_voltages_file = 'input_voltages1.csv'
data_file_csv = 'double_9x18_crossbar.csv'
data_file_raw = 'double_9x18_crossbar.raw'
bias_value = 0.1
learning_rate = 0.01
epochs = 5
batches = 5
columns_to_plot = ['V(out00)', 'V(out01)', 'V(out02)', 'V(out03)', 'V(out04)', 'V(out05)', 'V(out06)', 'V(out07)',
                   'V(out08)']
sim = MemCrossbarSim(netlist_file, input_voltages_file, data_file_raw, data_file_csv,
                     bias_value=bias_value, learning_rate=learning_rate, epochs=epochs, batch_size=batches)
sim.train()
print("Training complete.")
sim.plot_losses()
sim.plot_specified_columns(columns_to_plot)

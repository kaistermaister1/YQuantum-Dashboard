import glob
import json
import math
import os
import re

import gurobipy as gp
import numpy as np
from gurobipy import GRB, Model


def run_gurobi(json_file_path):
    """
    Function that runs the Gurobi optimization on an ILP instance.
    Takes as input the path to a JSON file containing the specifications of the objective
    and constraints of an ILP instance.

    Inputs:
    json_file_path: str
       Path to JSON file containing specifications for an ILP instance.


    Outputs:
    Returns the optimised objective value, accessed as an attribute of the model from gurobipy.

    model.objVal: float
        Optimised objective value
    """
    # Load the dataset from the JSON structure
    with open(json_file_path) as f:
        data = json.load(f)

    # Create a new model
    model = gp.Model("MILP_Optimization")

    # Define decision variables
    num_vars = len(data["objective"])
    x = model.addVars(num_vars, vtype=GRB.BINARY, name="x")

    # Set the objective function
    model.setObjective(
        gp.quicksum(data["objective"][i] * x[i] for i in range(num_vars)),
        GRB.MAXIMIZE,
    )

    # Add constraints
    for j in range(len(data["constraints"])):
        model.addConstr(
            gp.quicksum(data["constraints"][j][i] * x[i] for i in range(num_vars))
            <= data["constraints_rhs"][j],
            f"Constraint_{j}",
        )

    # Optimize the model
    model.optimize()

    # Print the results
    if model.status == GRB.OPTIMAL:
        print("Optimal objective value:", model.objVal)
        for i in range(num_vars):
            print(f"x[{i}] =", x[i].X)
    else:
        print("No optimal solution found.")

    return model.objVal


def transform_xor_problems_for_gurobi(
    npz_folder,
    out_folder,
    pattern="iter_*.npz",
):
    """Convert saved Max-Xorsat matrices (B,v) into XOR-equation text files for Gurobi.

    Scans `.npz` files in `npz_folder` matching `pattern`, extracts the
    matrix B and RHS vector v, and writes each instance as a `.txt` file
    in `out_folder`. Returns the list of B shapes and matrices.
    """
    # make sure output dir exists
    os.makedirs(out_folder, exist_ok=True)

    b_shape_list = []
    b_list = []

    # find all saved instances
    files = sorted(glob.glob(os.path.join(npz_folder, pattern)))
    for fn in files:
        # recover iteration index from filename, e.g. "iter_005.npz" → 5
        base = os.path.basename(fn)
        try:
            idx = int(base.split("_")[1].split(".")[0])
        except ValueError:
            print(f"Skipping unrecognized file name: {fn}")
            continue

        # load B and v
        data = np.load(fn)
        B = data["B"]
        # print(B)
        b_shape_list.append(B.shape)
        b_list.append(B)
        v = data["v"].astype(int)  # ensure plain Python ints

        # prepare output txt file
        out_fn = os.path.join(out_folder, f"iter_{idx}.txt")
        with open(out_fn, "w") as f:
            # each row of B becomes one XOR equation
            for row_idx, (row, rhs) in enumerate(zip(B, v)):
                # collect variable names where row[j] == 1
                vars_in_eq = [f"x{j}" for j, val in enumerate(row) if val == 1]
                if not vars_in_eq:
                    # (optional) handle all-zero rows
                    line = f"0 = {rhs}\n"
                else:
                    line = " XOR ".join(vars_in_eq) + f" = {rhs}\n"
                f.write(line)

        # print(f"Wrote: {out_fn}")

    return b_shape_list, b_list


def solve_max_xor_sat(data, filename, filename_save):
    """Solve a Max-XOR-SAT problem instance with Gurobi.

    Args:
        data (list[str]): Lines describing XOR constraints in the form
            "x1 XOR x2 XOR ... = {0,1}".
        filename (str): Name suffix for saving the solution results
            in the `results_max_xor_sat/` folder.

    Behavior:
        - Creates binary decision variables for all encountered symbols.
        - Encodes each XOR constraint using auxiliary variables and
          a satisfaction indicator.
        - Maximizes the number of satisfied XOR constraints.
        - Prints variable assignments and objective value if optimal.
        - Writes results (objective value, ratio, assignments, and selected
          variables) to `solution_<filename>`.

    Returns:
        None
    """
    model = Model("Max-XOR-SAT")

    # Create a dictionary to hold the variables
    variables = {}
    sat_variables = {}
    constraint_count = 0
    # Parse the dataset
    for line in data:
        # Split the line into parts
        parts = line.split("=")
        if len(parts) == 2:
            # If the line contains a variable assignment
            cons, value = parts[0].strip(), parts[1].strip()
            if value in ["0", "1"]:
                xor_vars = cons.split("XOR")
                xor_vars = [var.strip() for var in xor_vars]
                for var in xor_vars:
                    # Create a binary variable for each XOR variable
                    if var not in variables:
                        variables[var] = model.addVar(vtype=GRB.BINARY, name=var)
                # Add the XOR constraint
                # The sum of the variables in an XOR must be odd if the value is 1 and even if the value is 0.
                aux_var = model.addVar(
                    vtype=GRB.INTEGER,
                    lb=-float("inf"),
                    name="aux_var",
                )
                sat_variables["sat_var_" + str(constraint_count)] = model.addVar(
                    vtype=GRB.BINARY,
                    name="sat_var_" + str(constraint_count),
                )
                model.addConstr(
                    gp.quicksum(variables[var] for var in xor_vars)
                    == 1
                    + 2 * aux_var
                    + (int(value) - sat_variables["sat_var_" + str(constraint_count)]),
                    "aux_var_constraint_" + str(constraint_count),
                )
                constraint_count += 1
            else:
                print(f"Unexpected value: {value}. Expected 0 or 1.")
                continue
        else:
            print(f"Unexpected line: {line}")
            continue

    # Define the objective function: maximize the sum of satisfied XOR constraints
    model.setObjective(sum(sat_variables[var] for var in sat_variables), GRB.MAXIMIZE)

    # Optimize the model
    model.optimize()

    # Print the results
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found:")
        for var in variables:
            print(f"{var}: {variables[var].X}")
        print(f"Objective value: {model.objVal}")
    else:
        print("No optimal solution found.")

    os.makedirs(os.path.dirname(filename_save), exist_ok=True)
    with open(filename_save, "w") as f:
        if model.status == GRB.OPTIMAL:
            selected_variables = []
            f.write(
                "Optimal objective value (number of satisfied XOR constraints): "
                + str(model.objVal)
                + "\n",
            )
            f.write("Number of total constraints: " + str(constraint_count) + "\n")
            f.write(
                "Ratio of satisfied constraints: "
                + str((model.objVal / constraint_count) * 100)
                + "%"
                + "\n",
            )
            for i in sorted(variables.keys()):
                f.write(f"{i}: {variables[i].X}\n")
                if variables[i].X > 0.5:
                    selected_variables.append(i)
            f.write("Selected variables: " + str(selected_variables) + "\n")
        else:
            f.write("No optimal solution found.\n")


def s(B, v, x):
    """
    Computes the number of satisfied parity-check constraints for a given bitstring x.

    For each row of the parity-check matrix B, checks whether the linear constraint
    Bᵢ·x ≡ vᵢ (mod 2) is satisfied. Returns the total count of satisfied constraints.

    Args:
        B (np.ndarray): Binary parity-check matrix (shape m x n).
        v (np.ndarray): Binary vector of length m representing the syndrome or constraint targets.
        x (Union[str, List[int]]): Bitstring (as a string of '0'/'1' or list of ints) representing a candidate solution.

    Returns:
        int: Number of satisfied constraints (between 0 and m).

    Raises:
        ValueError: If the bitstring format is invalid or does not match the expected length.
    """
    # Convert bitstring to list of ints if needed
    if isinstance(x, str):
        if not all(c in "01" for c in x):
            raise ValueError("Bitstring must contain only '0' and '1'")
        x = [int(bit) for bit in x]

    num_constrains = B.shape[0]
    num_variables = B.shape[1]

    if num_variables != len(x):
        raise ValueError(
            f"Mismatch: expected {num_variables} variables, but got {len(x)} values in x.",
        )

    satisfied_constrains = 0
    for constrain in range(num_constrains):
        v_op = 0
        for var in range(num_variables):
            v_op = (v_op + B[constrain, var] * x[var]) % 2

        if v[constrain] == v_op:
            satisfied_constrains += 1

    return satisfied_constrains


def read_gurobi_results_fun(path_gurobi, path_B, pattern="iter_*.npz"):
    """
    Reads Gurobi optimization results and corresponding binary matrices.

    For each `.npz` file matching the pattern in `path_B`, this function:
    - Extracts iteration index and loads the matrix B, vector v, and shape.
    - Locates and parses the corresponding `solution_iter_k.txt` file from `path_gurobi`:
        - Extracts objective value, constraint satisfaction ratio, and selected variables.
        - Builds a bitstring where selected variables are marked as 1s.
    - Recomputes the number of satisfied constraints using a user-defined `s(B, v, bitstring)` function.
    - Validates that the satisfaction ratio matches the parsed value from the solution file.
      If not, raises a `ValueError`.

    Returns:
        A list of tuples for each valid problem instance:
        (iteration_index, B, v, shape_B, filename, solution_info_dict)

    Raises:
        ValueError: if satisfaction ratio from file and computed ratio do not match.
    """
    files = sorted(glob.glob(os.path.join(path_B, pattern)))
    problem_instances = []

    for fn in files:
        # Extract iteration index, e.g., iter_005.npz -> 5
        base = os.path.basename(fn)
        try:
            idx = int(base.split("_")[1].split(".")[0])
        except Exception:
            idx = None

        # Load the NPZ file
        try:
            data = np.load(fn)
            B = data["B"]
            v = data["v"]
            shape_B = tuple(data["shape_B"])
        except Exception as e:
            print(f"Warning: could not load '{fn}': {e}")
            continue

        # Load corresponding solution file
        solution_info = {}
        solution_filename = path_gurobi + f"/iter_{idx}.txt"
        if os.path.exists(solution_filename):
            with open(solution_filename, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "Optimal objective value" in line:
                        solution_info["objective_value"] = float(
                            re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)[0],
                        )
                    elif "Number of total constraints" in line:
                        solution_info["total_constraints"] = int(
                            re.findall(r"\d+", line)[0],
                        )
                    elif "Ratio of satisfied constraints" in line:
                        solution_info["satisfaction_ratio"] = float(
                            re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)[0],
                        )
                    elif line.startswith("x["):
                        m = re.match(r"x\[(\d+)\]\s*=\s*([01]\.0)", line)
                        if m:
                            if "x" not in solution_info:
                                solution_info["x"] = {}
                            solution_info["x"][int(m.group(1))] = float(m.group(2))
                    elif line.startswith("Selected variables:"):
                        # Handle both formats
                        if "x" in line:
                            # Case: ['x10', 'x12', ...] → extract digits after 'x'
                            selected = [int(s[1:]) for s in re.findall(r"x\d+", line)]
                        else:
                            # Case: [0, 1, 2, ...]
                            selected = list(map(int, re.findall(r"\d+", line)))
                        solution_info["selected_vars"] = selected

            s_average = solution_info.get("satisfaction_ratio")
            selected_variables = solution_info.get("selected_vars")

            n_vars = B.shape[1]  # or shape_B[1]

            # Create a bitstring list initialized to 0
            bitstring = [0] * n_vars
            # Set 1s at selected variable positions
            if selected_variables:  # skip if it is none
                for i in selected_variables:
                    bitstring[i] = 1

            solution_btsr = "".join(map(str, bitstring))

            satisfied = s(B, v, solution_btsr)

            satisfied_percentage = satisfied * 100 / B.shape[0]

            if not math.isclose(s_average, satisfied_percentage, rel_tol=1e-9):
                raise ValueError(
                    f"Mismatch in satisfaction ratio: file has {s_average}, computed {satisfied_percentage}",
                )

            # Append the combined data
            problem_instances.append((idx, B, v, shape_B, fn, solution_info))

    return problem_instances


if __name__ == "__main__":
    json_file_path = "classical_pipeline/data/milp_formulation.json"
    gurobi_objective = run_gurobi(json_file_path)

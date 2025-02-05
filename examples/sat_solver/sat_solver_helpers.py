import os
import random

from solvers.simple_solver.simple_sat import find_all_solutions, save_solutions_to_file
from solvers.simple_solver import iterative, recursive

# Global counter for output file names
counter = 0

# Base directory relative to the script's location
base_dir = os.path.dirname(os.path.abspath(__file__))


def generate_random_cnf(num_vars, num_clauses, num_files, output_dir):
    """
    Generate random CNF files with a specified number of variables and clauses,
    without a CNF header and without trailing zeros.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_paths = []

    for i in range(num_files):
        file_path = os.path.join(output_dir, f"input_cnf_{i + 1}.cnf")
        with open(file_path, "w") as f:
            for _ in range(num_clauses):
                # Create a list of variables (1 to num_vars)
                variables = list(range(1, num_vars + 1))

                # Shuffle variables and apply negation randomly
                random.shuffle(variables)
                literals = [
                    f"~{var}" if random.choice([True, False]) else str(var)
                    for var in variables
                ]

                # Join literals to form a clause and write to file
                clause_str = " ".join(literals)
                f.write(clause_str + "\n")

        file_paths.append(file_path)

    return file_paths


def read_cnf_file(cnf_file_path):
    """
    Reads a CNF file and returns a list of clauses and a set of existing variables
    with their negation status.
    """
    existing_variables = set()
    clauses = []

    with open(cnf_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("c"):
                literals = line.split()
                clause = []
                for lit in literals:
                    if lit != "0":
                        negated = lit.startswith("~")
                        var = int(lit.lstrip("~"))
                        existing_variables.add((var, negated))
                        clause.append((var, negated))
                clauses.append(clause)

    return clauses, existing_variables


def write_cnf_file(output_file_path, clauses):
    """
    Writes a list of clauses to a CNF file.
    """
    with open(output_file_path, "w") as f:
        for clause in clauses:
            clause_str = " ".join(
                f"~{var}" if negated else str(var) for var, negated in clause
            )
            f.write(clause_str + "\n")


def parse_solution_file(file_path):
    """
    Parses the SAT solution file and converts it to a solution array.
    Interprets '~' as 0 and non '~' as 1.
    """
    solution_array = []

    # Open the file and read lines
    with open(file_path, "r") as file:
        for line in file:
            # Remove any leading/trailing whitespace
            line = line.strip()
            if line:
                # Split the line into literals
                literals = line.split()
                # Convert literals to the corresponding binary array
                solution = [0 if lit.startswith("~") else 1 for lit in literals]
                solution_array.append(solution)

    return solution_array


def get_unique_output_path(base_path):
    global counter
    counter += 1
    return f"{base_path}_{counter}.txt"


def get_followup_file_path(file_name):
    return os.path.join(base_dir, "test_data", file_name)


def run_solver_and_return_path(cnf_file_path, algorithm=None):
    """
    Runs the SAT solver with the specified algorithm and returns a unique output file path.
    """
    output_dir = os.path.join(base_dir, "test_data")

    os.makedirs(output_dir, exist_ok=True)

    output_file_path = os.path.join(output_dir, "output")
    output_file_path = get_unique_output_path(output_file_path)

    # Choose the solver based on the algorithm argument
    solver = iterative if algorithm == "iterative" else recursive

    # Call the solver
    with open(cnf_file_path, "r") as input_file:
        solutions = find_all_solutions(input_file, solver, verbose=False)

    save_solutions_to_file(solutions, output_file_path)

    return output_file_path

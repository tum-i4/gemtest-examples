from sat_solver_helpers import (
    generate_random_cnf,
    read_cnf_file,
    write_cnf_file,
    parse_solution_file,
    run_solver_and_return_path,
    get_followup_file_path
)


import gemtest as gmt
import random
import os

# Base directory relative to the script's location
base_dir = os.path.dirname(os.path.abspath(__file__))

num_vars = random.randint(1,10)
num_clauses = 1
num_files = 5

# Base directory relative to the script's location
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "test_data")

os.makedirs(output_dir, exist_ok=True)

# Generate the files
single_clause_cnf_file_paths = generate_random_cnf(
    num_vars, num_clauses, num_files, output_dir
)

# Create the metamorphic relations
new_disjunction_single = gmt.create_metamorphic_relation(
    name="new_disjunction_single", data=single_clause_cnf_file_paths
)
new_negated_disjunction_single = gmt.create_metamorphic_relation(
    name="new_negated_disjunction_single", data=single_clause_cnf_file_paths
)
disjunction_duplicate_variable = gmt.create_metamorphic_relation(
    name="disjunction_duplicate_variable", data=single_clause_cnf_file_paths
)
tautology = gmt.create_metamorphic_relation(
    name="tautology", data=single_clause_cnf_file_paths
)
conjunction_with_new_clause = gmt.create_metamorphic_relation(
    name="conjunction_with_new_clause", data=single_clause_cnf_file_paths
)


@gmt.transformation(new_disjunction_single)
def add_new_variable_disjunction(cnf_file_path):
    """
    Adds a new variable as a disjunction to the clause in the CNF file and writes the result to a fixed output file.
    """
    clauses, existing_variables = read_cnf_file(cnf_file_path)
    new_variable = max((var for var, _ in existing_variables)) + 1

    updated_clauses = [clause + [(new_variable, False)] for clause in clauses]

    output_file_path = get_followup_file_path("input_transformation_1.cnf")
    write_cnf_file(output_file_path, updated_clauses)

    return output_file_path


@gmt.transformation(new_negated_disjunction_single)
def add_new_negated_variable_disjunction(cnf_file_path):
    """
    Adds a new variable as a negated disjunction to each clause in the CNF file and writes the result to a fixed output file.
    """
    clauses, existing_variables = read_cnf_file(cnf_file_path)
    new_variable = max((var for var, _ in existing_variables)) + 1

    # Add new variable as negated disjunction to each clause
    updated_clauses = [clause + [(new_variable, True)] for clause in clauses]

    output_file_path = get_followup_file_path("input_transformation_2.cnf")
    write_cnf_file(output_file_path, updated_clauses)

    return output_file_path


@gmt.transformation(disjunction_duplicate_variable)
def add_existing_variable_disjunction(cnf_file_path):
    """
    Adds an existing variable as a disjunction to each clause in the CNF file and writes the result to a fixed output file.
    If the original variable is negated, the duplicate variable will also be negated.
    """
    clauses, existing_variables = read_cnf_file(cnf_file_path)

    # Randomly select an existing variable and its negation status
    existing_variable, is_negated = random.choice(list(existing_variables))

    # Add the selected existing variable as disjunction to each clause
    updated_clauses = [clause + [(existing_variable, is_negated)] for clause in clauses]

    output_file_path = get_followup_file_path("input_transformation_3.cnf")
    write_cnf_file(output_file_path, updated_clauses)

    return output_file_path


@gmt.transformation(tautology)
def add_existing_variable_negation_disjunction(cnf_file_path):
    """
    Adds an existing variable as both a negation and a disjunction to each clause in the CNF file
    and writes the result to a fixed output file.
    """
    clauses, existing_variables = read_cnf_file(cnf_file_path)

    # Randomly select an existing variable
    existing_variable = random.choice(list(existing_variables))[0]

    # Add the selected existing variable and its negation to each clause
    updated_clauses = [
        clause + [(existing_variable, False), (existing_variable, True)]
        for clause in clauses
    ]

    output_file_path = get_followup_file_path("input_transformation_4.cnf")
    write_cnf_file(output_file_path, updated_clauses)

    return output_file_path


@gmt.transformation(conjunction_with_new_clause)
def add_conjunction_with_new_clause(cnf_file_path):
    """
    Adds a new clause (involving only existing variables) to an existing CNF file.
    The new clause is put into conjunction with a single original clause.
    """
    clauses, existing_variables = read_cnf_file(cnf_file_path)

    if not clauses:
        raise ValueError("No clauses found in the CNF file.")

    # Select a random existing clause to add
    new_clause = random.choice(clauses)

    # Add the selected clause to the end
    updated_clauses = clauses + [new_clause]

    output_file_path = get_followup_file_path("input_transformation_5.cnf")
    write_cnf_file(output_file_path, updated_clauses)

    return output_file_path


@gmt.relation(new_disjunction_single, new_negated_disjunction_single)
def verify_assignment_amount(source_output, followup_output):
    """For single clause CNFs the expected amount of assignments satisfying the formula is 2 times the original
    number of assignments + 1. (Original Assignments with new variable set to 0 and 1 each plus
    False assignment with new variable = 1)
    """
    source_solution = parse_solution_file(source_output)
    followup_solution = parse_solution_file(followup_output)
    original_count = len(source_solution)
    transformed_count = len(followup_solution)
    expected_count = 2 * original_count + 1

    # Check if each assignment in source_output is correctly extended in followup_output
    correct_extension = all(
        assignment + [0] in followup_solution and assignment + [1] in followup_solution
        for assignment in source_solution
    )

    return correct_extension and (transformed_count == expected_count)


@gmt.relation(disjunction_duplicate_variable)
def equality(source_output, followup_output):
    source_solution = parse_solution_file(source_output)
    followup_solution = parse_solution_file(followup_output)
    return source_solution == followup_solution


@gmt.relation(tautology)
def equality_length(source_output, followup_output):
    source_solution = parse_solution_file(source_output)
    followup_solution = parse_solution_file(followup_output)
    number_vars = len(source_solution[0])

    expected_num_solutions = 2**number_vars

    return len(followup_solution) == expected_num_solutions


@gmt.relation(conjunction_with_new_clause)
def subset(source_output, followup_output):
    source_solution = parse_solution_file(source_output)
    followup_solution = parse_solution_file(followup_output)
    return all(assignment in followup_solution for assignment in source_solution)


@gmt.system_under_test()
def test_recursive(cnf_file_path):
    return run_solver_and_return_path(cnf_file_path, algorithm="recursive")


@gmt.system_under_test()
def test_iterative(cnf_file_path):
    return run_solver_and_return_path(cnf_file_path, algorithm="iterative")
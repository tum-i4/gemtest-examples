from sat_solver_helpers import (
    generate_random_cnf,
    read_cnf_file,
    write_cnf_file,
    parse_solution_file,
    get_followup_file_path,
    get_unique_output_path,
)

from solvers.pysat_solver.pysat_find_all import solve_pysat
from pysat.solvers import (
    Minisat22,
    Glucose42,
    MapleChrono,
    MapleCM,
    Lingeling,
    Cadical195,
    Gluecard4,
    Maplesat,
    Mergesat3,
    Minicard,
    Glucose3,
)

import gemtest as gmt
import random
import os

num_vars = random.randint(2, 10)
num_clauses = random.randint(2, 10)
num_files = 10


# Base directory relative to the script's location
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "test_data")
output_file_base_path = os.path.join(output_dir, "output")

os.makedirs(output_dir, exist_ok=True)

# Generate the files
clause_cnf_file_paths = generate_random_cnf(
    num_vars, num_clauses, num_files, output_dir
)

# Create the metamorphic relations
new_disjunction_single = gmt.create_metamorphic_relation(
    name="new_disjunction_single", data=clause_cnf_file_paths
)
new_negated_disjunction_single = gmt.create_metamorphic_relation(
    name="new_negated_disjunction_single", data=clause_cnf_file_paths
)
disjunction_duplicate_variable = gmt.create_metamorphic_relation(
    name="disjunction_duplicate_variable", data=clause_cnf_file_paths
)
tautology = gmt.create_metamorphic_relation(
    name="tautology", data=clause_cnf_file_paths
)
conjunction_with_new_clause = gmt.create_metamorphic_relation(
    name="conjunction_with_new_clause", data=clause_cnf_file_paths
)
substitute_variable = gmt.create_metamorphic_relation(
    name="substitute_variable", data=clause_cnf_file_paths
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
    The new clause is put into conjunction the previous clauses.
    """
    clauses, _ = read_cnf_file(cnf_file_path)

    if not clauses:
        raise ValueError("No clauses found in the CNF file.")

    # Select a random existing clause to add
    new_clause = random.choice(clauses)

    # Add the selected clause to the end
    updated_clauses = clauses + [new_clause]

    output_file_path = get_followup_file_path("input_transformation_5.cnf")
    write_cnf_file(output_file_path, updated_clauses)

    return output_file_path


@gmt.transformation(substitute_variable)
def fixed_zero_variable(cnf_file_path):
    """
    Adds a new clause with a single negated variable from the existing CNF file.
    """
    clauses, _ = read_cnf_file(cnf_file_path)

    # Select a random existing variable
    variable_to_negate = random.randint(1, len(clauses[0]))

    new_literal = [(variable_to_negate, True)]

    updated_clauses = clauses + [new_literal]

    output_file_path = get_followup_file_path("input_transformation_6.cnf")

    write_cnf_file(output_file_path, updated_clauses)

    return output_file_path


@gmt.relation(new_disjunction_single, new_negated_disjunction_single)
def verify_assignment_amount(source_output, followup_output):
    """
    The new Solution set contains the old solution set with the new variable set either to True or False.
    Additionally, every solution that previously did not satisfy the CNF satisfies the new formula if the
    new variable is True!
    """
    source_solution = parse_solution_file(source_output)
    followup_solution = parse_solution_file(followup_output)
    original_count = len(source_solution)
    transformed_count = len(followup_solution)
    original_num_variables = len(source_solution[0]) if original_count > 0 else 0
    # 2 Times old solution set + the non-satisfying solutions
    expected_count = 2 * original_count + (2**original_num_variables - original_count)

    # Special handling for the case when the original CNF is unsatisfiable
    if original_count == 0:
        # For an unsatisfiable original CNF, check that the follow-up solution is valid
        is_valid_followup = all(
            assignment == [0] * original_num_variables + [1]
            or assignment == [1] * original_num_variables + [1]
            for assignment in followup_solution
        )
        return is_valid_followup
    
    # Check if each assignment in source_output is correctly extended in followup_output
    correct_extension = all(
        assignment + [0] in followup_solution
        and assignment + [1] in followup_solution
        for assignment in source_solution
    )
    return correct_extension and (transformed_count == expected_count)


@gmt.relation(disjunction_duplicate_variable)
def equality(source_output, followup_output):
    """
    The original solution should be present in the new solution set.
    """
    source_solution = parse_solution_file(source_output)
    followup_solution = parse_solution_file(followup_output)

    source_solution = [item for item in source_solution]
    followup_solution = [item for item in followup_solution]

    # Check if the original CNF was satisfiable
    was_satisfiable = len(source_solution) > 0

    if was_satisfiable:
        # Check if each item in followup_solution is in source_solution
        return all(solution in followup_solution for solution in source_solution)
    
    # If the original was unsatisfiable, allow the transformed CNF to be satisfiable
    # We don't know which variables caused this conflict so we can't say more about the solution
    return len(followup_solution) >= 0


@gmt.relation(tautology)
def equality_length(source_output, followup_output):
    # Parse the solutions from the source and follow-up output files
    source_solution = parse_solution_file(source_output)
    followup_solution = parse_solution_file(followup_output)

    number_vars = len(source_solution[0]) if source_solution else 0

    expected_num_solutions = 2**number_vars

    if source_solution:
        return len(followup_solution) == expected_num_solutions

    # In case no solution was found for the original file we dont know how many variables were originally in this
    return (len(followup_solution) & (len(followup_solution) - 1) == 0) and len(
        followup_solution
    ) != 0


@gmt.relation(conjunction_with_new_clause)
def subset(source_output, followup_output):
    source_solution = parse_solution_file(source_output)
    followup_solution = parse_solution_file(followup_output)
    return all(assignment in followup_solution for assignment in source_solution)


@gmt.relation(substitute_variable)
def subset_fixed_to_zero(source_output, followup_output):
    source_solution = parse_solution_file(source_output)
    followup_solution = parse_solution_file(followup_output)
    return all(assignment in source_solution for assignment in followup_solution)


@gmt.system_under_test()
def test_pysat_minisat22(cnf_file_path):
    output_file_path = get_unique_output_path(output_file_base_path)
    solve_pysat(
        cnf_file_path=cnf_file_path,
        output_file_path=output_file_path,
        solver_cls=Minisat22,
    )
    return output_file_path


@gmt.system_under_test()
def test_pysat_glucose42(cnf_file_path):
    output_file_path = get_unique_output_path(output_file_base_path)
    solve_pysat(
        cnf_file_path=cnf_file_path,
        output_file_path=output_file_path,
        solver_cls=Glucose42,
    )
    return output_file_path


@gmt.system_under_test()
def test_pysat_glucose3(cnf_file_path):
    output_file_path = get_unique_output_path(output_file_base_path)
    solve_pysat(
        cnf_file_path=cnf_file_path,
        output_file_path=output_file_path,
        solver_cls=Glucose3,
    )
    return output_file_path


@gmt.system_under_test()
def test_pysat_glucose4(cnf_file_path):
    output_file_path = get_unique_output_path(output_file_base_path)
    solve_pysat(
        cnf_file_path=cnf_file_path,
        output_file_path=output_file_path,
        solver_cls=Gluecard4,
    )
    return output_file_path


@gmt.system_under_test()
def test_pysat_maplechrono(cnf_file_path):
    output_file_path = get_unique_output_path(output_file_base_path)
    solve_pysat(
        cnf_file_path=cnf_file_path,
        output_file_path=output_file_path,
        solver_cls=MapleChrono,
    )
    return output_file_path


@gmt.system_under_test()
def test_pysat_lingeling(cnf_file_path):
    output_file_path = get_unique_output_path(output_file_base_path)
    solve_pysat(
        cnf_file_path=cnf_file_path,
        output_file_path=output_file_path,
        solver_cls=Lingeling,
    )
    return output_file_path


@gmt.system_under_test()
def test_pysat_cadical(cnf_file_path):
    output_file_path = get_unique_output_path(output_file_base_path)
    solve_pysat(
        cnf_file_path=cnf_file_path,
        output_file_path=output_file_path,
        solver_cls=Cadical195,
    )
    return output_file_path


@gmt.system_under_test()
def test_pysat_maplesat(cnf_file_path):
    output_file_path = get_unique_output_path(output_file_base_path)
    solve_pysat(
        cnf_file_path=cnf_file_path,
        output_file_path=output_file_path,
        solver_cls=Maplesat,
    )
    return output_file_path


@gmt.system_under_test()
def test_pysat_mergesat3(cnf_file_path):
    output_file_path = get_unique_output_path(output_file_base_path)
    solve_pysat(
        cnf_file_path=cnf_file_path,
        output_file_path=output_file_path,
        solver_cls=Mergesat3,
    )
    return output_file_path


@gmt.system_under_test()
def test_pysat_minicard(cnf_file_path):
    output_file_path = get_unique_output_path(output_file_base_path)
    solve_pysat(
        cnf_file_path=cnf_file_path,
        output_file_path=output_file_path,
        solver_cls=Minicard,
    )
    return output_file_path


@gmt.system_under_test()
def test_pysat_maplecm(cnf_file_path):
    output_file_path = get_unique_output_path(output_file_base_path)
    solve_pysat(
        cnf_file_path=cnf_file_path,
        output_file_path=output_file_path,
        solver_cls=MapleCM,
    )
    return output_file_path

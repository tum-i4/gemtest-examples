import os
from pysat.formula import CNF
from pysat.solvers import Minisat22


def parse_cnf_file(cnf_file_path):
    """
    Parse a CNF file and return a list of clauses in the format required by pysat.
    """
    clauses = []

    with open(cnf_file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                literals = line.split()
                clause = []
                for lit in literals:
                    if lit == "0":
                        break
                    elif lit.startswith("~"):
                        var = int(lit[1:])
                        clause.append(-var)
                    else:
                        var = int(lit)
                        clause.append(var)
                if clause:
                    clauses.append(clause)

    return clauses


def write_solutions_to_cnf_file(output_file_path, solutions):
    """
    Writes the satisfying assignments to a CNF file.
    """
    with open(output_file_path, "w") as f:
        for solution in solutions:
            # Convert literals to the required format and join them into a single string
            clause_str = " ".join(
                f"~{abs(lit)}" if lit < 0 else str(lit) for lit in solution
            )
            f.write(clause_str + "\n")


def solve_pysat(cnf_file_path, output_file_path, solver_cls=Minisat22):
    """
    Load a CNF file, parse it, and find all solutions using the specified solver.
    """
    # Parse CNF file
    clauses = parse_cnf_file(cnf_file_path)

    # Create CNF object for pysat
    formula = CNF()
    formula.extend(clauses)

    solutions = []

    # Create solver instance using the passed solver class
    with solver_cls(bootstrap_with=formula) as solver:
        while solver.solve():
            model = solver.get_model()
            solutions.append(model)

            # Block the current solution by negating and appending as a clause
            blocked_clause = [-lit for lit in model]
            solver.add_clause(blocked_clause)

    write_solutions_to_cnf_file(output_file_path, solutions)

    return output_file_path, solutions

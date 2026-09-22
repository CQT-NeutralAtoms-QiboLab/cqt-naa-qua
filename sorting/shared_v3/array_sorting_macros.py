from qm.qua import *


def get_current_row(
    this_row,
    nb_of_cols,
    current_location_full,
    target_location_full,
    target_frequencies_full,
):
    current_location = declare(int, size=nb_of_cols)
    target_location = declare(int, size=nb_of_cols)
    target_frequencies = declare(int, size=nb_of_cols)
    freq_count = declare(int)
    assign(freq_count, 0)
    col = declare(int)
    with for_(col, 0, col < nb_of_cols, col + 1):
        assign(current_location[col], current_location_full[this_row * nb_of_cols + col])
        assign(target_location[col], target_location_full[this_row * nb_of_cols + col])
        with if_(~(target_frequencies_full[this_row * nb_of_cols + col] == 0)):
            assign(
                target_frequencies[freq_count],
                target_frequencies_full[this_row * nb_of_cols + col],
            )
            assign(freq_count, freq_count + 1)
    return current_location, target_location, target_frequencies


def find_number_of_tweezers(atoms_in_current_row, atoms_in_target_row, max_nb_of_tweezers):
    nb_of_tweezers = declare(int)
    atoms_or_targets = declare(int, size=3)
    assign(atoms_or_targets[0], Math.sum(atoms_in_current_row))
    assign(atoms_or_targets[1], Math.sum(atoms_in_target_row))
    assign(atoms_or_targets[2], max_nb_of_tweezers)
    assign(nb_of_tweezers, Math.min(atoms_or_targets))
    return nb_of_tweezers


def assign_tweezers_to_atoms(
    nb_of_tweezers,
    nb_of_tweezers_python,
    atoms_in_current_row,
    current_frequencies,
    target_frequencies,
    nb_of_cols,
):
    """Left-to-right row compression: first N occupied sites -> first N targets."""
    frequencies = declare(int, size=nb_of_tweezers_python)
    detunings = declare(int, size=nb_of_tweezers_python)
    i = declare(int)
    j = declare(int)

    assign(i, 0)
    with for_(j, 0, j < nb_of_tweezers, j + 1):
        with while_((i < nb_of_cols) & (atoms_in_current_row[i] == 0)):
            assign(i, i + 1)
        with if_(i < nb_of_cols):
            assign(frequencies[j], current_frequencies[i])
            assign(detunings[j], target_frequencies[j] - frequencies[j])
            assign(i, i + 1)

    return frequencies, detunings


def print_2d(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(f"{matrix[i][j]}\t", end="")
        print("")


def python_row_compression_final(atom_location_list, atom_target_list, max_tweezers):
    """Python mirror of assign_tweezers_to_atoms for occupation-matrix plots."""
    atom_final_list = []
    for loc_row, tgt_row in zip(atom_location_list, atom_target_list):
        loc_row = list(loc_row)
        tgt_row = list(tgt_row)
        atom_idx = [i for i, v in enumerate(loc_row) if v]
        target_idx = [i for i, v in enumerate(tgt_row) if v]
        n = min(len(atom_idx), len(target_idx), max_tweezers)
        final = [0] * len(loc_row)
        for t in target_idx[:n]:
            final[t] = 1
        for leftover in atom_idx[n:]:
            final[leftover] = 1
        atom_final_list.append(final)
    return atom_final_list

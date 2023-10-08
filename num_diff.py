from typing import List, Callable, SupportsFloat as Numeric

def numdiff_central(inputs: List[Numeric], h: Numeric, func: Callable):
    r"""
            numdiff_central(inputs, h, func)

            Computes central finite differences for each input in inputs

            inputs:
                inputs: List of float numbers representing linear space over specified interval
                h: size interval
                func: function which derivative is computing
    """

    value_table = []
    for x in inputs:
        value_table.append([x, func(x)])
    diff_list = []

    for i in range(len(inputs)):
        if i == 0:
            diff_list.append((-3 * value_table[0][1] + 4 * value_table[1][1] - value_table[2][1]) / (2 * h))
        elif i == len(inputs) - 1:
            diff_list.append((3 * value_table[i][1] - 4 * value_table[i - 1][1] + value_table[i - 2][1]) / (2 * h))
        else:
            diff_list.append((value_table[i + 1][1] - value_table[i - 1][1]) / (2 * h))

    return diff_list

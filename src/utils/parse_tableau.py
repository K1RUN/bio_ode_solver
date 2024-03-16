import fractions as fr


def parse_butcher_tableau() -> dict:
    """
    NEED TO SPECIFY BUTCHER TABLE FILE IN STDIN (default path is butcher_tables/...)
    :return: dict of lists with floats
    """
    path = 'butcher_tables/'
    filename = input()
    with open(path + filename) as file:
        info = file.read().splitlines()

    def is_valid_number(s: str):
        """CHECK IF STRING CONTAINS A VALID NUMBER: POSITIVE OR NEGATIVE RATIONAL (WITH '/' SYMBOL) OR INTEGER"""
        values = s.split('/')
        return (len(values) == 1 or len(values) == 2) and all(num.lstrip('-+').isdigit() for num in values)

    table = {}

    for i, line in enumerate(info):
        """LAST LINE IN INFO STANDS FOR b COEFFICIENTS, b COEFFICIENTS WILL BE ADDED SEPARATELY AFTER THE CYCLE"""
        """https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Implicit_Runge%E2%80%93Kutta_methods"""
        row = line.split()
        if all(is_valid_number(string) for string in row):
            row = [float(fr.Fraction(number)) for number in row]
            if i != len(info) - 1:
                table.setdefault('c_', []).append(row[0])
                table.setdefault('a_', []).append(row[1:])

            elif i == len(info) - 1:
                table.setdefault('b_', []).extend(row)

    return table

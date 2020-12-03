import pubchempy as pcp
import pandas as pd


def pubChemCompositionRetrival(compound_names):
    """
    Searches PubChem database to match compound names with input and return compound composition.
    :param compound_names:
    :return: returns a dataframe of compound names and their composition of C, H, N, O
    """
    compound_data = []
    size = len(compound_names)
    queuelength = size
    for compound_name in compound_names:
        print(str(queuelength) + " out of " + str(size) + " compounds left...")
        queuelength -= 1
        try:
            results = pcp.get_compounds(compound_name, 'name')
        except TimeoutError:
            print('Timeout Error')
        try:
            elements = results[0].elements
            numC = numH = numO = numN = 0
            for element in elements:
                if element == "C":
                    numC += 1
                elif element == "H":
                    numH += 1
                elif element == "N":
                    numN += 1
                elif element == "O":
                    numO += 1
            compound_data.append([compound_name, numC, numH, numN, numO])
        except IndexError:
            print("ERROR: No match in PubChem Database")

    df = pd.DataFrame(compound_data, columns=['Structure', 'C', 'H', 'N', 'O'])
    df.set_index('Structure', inplace=True)
    try:
        df.to_csv('../data/pubChemCompositionRetrival_temp.csv')
    except FileNotFoundError:
        print('FileNotFoundError')
    return df

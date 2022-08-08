def levenshtein_distance(string1: str, string2: str):
    # computes the levenshtein distance between two strings
    lev = [[i + j for j in range(len(string2) + 1)] for i in range(len(string1) + 1)]

    for i in range(1, len(string1) + 1):
        for j in range(1, len(string2) + 1):
            if string1[i - 1] == string2[j - 1]:
                lev[i][j] = min(lev[i - 1][j] + 1, lev[i][j - 1] + 1, lev[i - 1][j - 1])
            else:
                lev[i][j] = min(lev[i - 1][j], lev[i][j - 1], lev[i - 1][j - 1]) + 1

    return lev[-1][-1]


def shared_term_similarity(list1: list, list2: list):
    # computes a similarity index between two lists of strings
    # each item in a list should represent a word
    s = 0
    for l1 in list1:
        if l1 in list2:
            s += 1

    s /= max(len(list1), len(list2))

    return s

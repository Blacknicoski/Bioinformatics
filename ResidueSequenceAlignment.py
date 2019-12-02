import numpy as np
from time import time

# set gap cost
first_cost = 2
repeat_cost = 0

# Building blosum62 matrix
blosum62 = [
    [9, -1, -1, -3, 0, -3, -3, -3, -4, -3, -3, -3, -3, -1, -1, -1, -1, -2, -2, -2],
    # C    Cys:[Cys:9, Ser:-1, Thr:-1, Pro:-3, Ala:0,  Gly:-3, Asn:-3, Asp:-3, Glu:-4, Gln:-3, His:-3, Arg:-3, Lys:-3, Met:-1, Ile:-1, Leu:-1, Val:-1, Phe:-2, Tyr:-2, Trp:-2]
    [-1, 4, 1, -1, 1, 0, 1, 0, 0, 0, -1, -1, 0, -1, -2, -2, -2, -2, -2, -3],
    # S   Ser:[Cys:-1,Ser:4,  Thr:1,  Pro:-1, Ala:1,  Gly:0,  Asn:1,  Asp:0,  Glu:0,  Gln:0,  His:-1, Arg:-1, Lys:0,  Met:-1, Ile:-2, Leu:-2, Val:-2, Phe:-2, Tyr:-2, Trp:-3]
    [-1, 1, 4, 1, -1, 1, 0, 1, 0, 0, 0, -1, 0, -1, -2, -2, -2, -2, -2, -3],
    # T    Thr:[Cys:-1,Ser:1,  Thr:4,  Pro:1,  Ala:-1, Gly:1,  Asn:0,  Asp:1,  Glu:0,  Gln:0,  His:0,  Arg:-1, Lys:0,  Met:-1, Ile:-2, Leu:-2, Val:-2, Phe:-2, Tyr:-2, Trp:-3]
    [-3, -1, 1, 7, -1, -2, -1, -1, -1, -1, -2, -2, -1, -2, -3, -3, -2, -4, -3, -4],
    # P    Pro:[Cys:-3,Ser:-1, Thr:1,  Pro:7,  Ala:-1, Gly:-2, Asn:-1, Asp:-1, Glu:-1, Gln:-1, His:-2, Arg:-2, Lys:-1, Met:-2, Ile:-3, Leu:-3, Val:-2, Phe:-4, Tyr:-3, Trp:-4]
    [0, 1, -1, -1, 4, 0, -1, -2, -1, -1, -2, -1, -1, -1, -1, -1, -2, -2, -2, -3],
    # A  Ala:[Cys:0, Ser:1,  Thr:-1, Pro:-1, Ala:4,  Gly:0,  Asn:-1, Asp:-2, Glu:-1, Gln:-1, His:-2, Arg:-1, Lys:-1, Met:-1, Ile:-1, Leu:-1, Val:-2, Phe:-2, Tyr:-2, Trp:-3]
    [-3, 0, 1, -2, 0, 6, -2, -1, -2, -2, -2, -2, -2, -3, -4, -4, 0, -3, -3, -2],
    # G   Gly:[Cys:-3,Ser:0,  Thr:1,  Pro:-2, Ala:0,  Gly:6,  Asn:-2, Asp:-1, Glu:-2, Gln:-2, His:-2, Arg:-2, Lys:-2, Met:-3, Ile:-4, Leu:-4, Val:0,  Phe:-3, Tyr:-3, Trp:-2]
    [-3, 1, 0, -2, -2, 0, 6, 1, 0, 0, -1, 0, 0, -2, -3, -3, -3, -3, -2, -4],
    # N   Asn:[Cys:-3,Ser:1,  Thr:0,  Pro:-2, Ala:-2, Gly:0,  Asn:6,  Asp:1,  Glu:0,  Gln:0,  His:-1, Arg:0,  Lys:0,  Met:-2, Ile:-3, Leu:-3, Val:-3, Phe:-3, Tyr:-2, Trp:-4]
    [-3, 0, 1, -1, -2, -1, 1, 6, 2, 0, -1, -2, -1, -3, -3, -4, -3, -3, -3, -4],
    # D    Asp:[Cys:-3,Ser:0,  Thr:1,  Pro:-1, Ala:-2, Gly:-1, Asn:1,  Asp:6,  Glu:2,  Gln:0,  His:-1, Arg:-2, Lys:-1, Met:-3, Ile:-3, Leu:-4, Val:-3, Phe:-3, Tyr:-3, Trp:-4]
    [-4, 0, 0, -1, -1, -2, 0, 2, 5, 2, 0, 0, 1, -2, -3, -3, -3, -3, -2, -3],
    # E   Glu:[Cys:-4,Ser:0,  Thr:0,  Pro:-1, Ala:-1, Gly:-2, Asn:0,  Asp:2,  Glu:5,  Gln:2,  His:0,  Arg:0,  Lys:1,  Met:-2, Ile:-3, Leu:-3, Val:-3, Phe:-3, Tyr:-2, Trp:-3]
    [-3, 0, 0, -1, -1, -2, 0, 0, 2, 5, 0, 1, 1, 0, -3, -2, -2, -3, -1, -2],
    # Q    Gln:[Cys:-3,Ser:0,  Thr:0,  Pro:-1, Ala:-1, Gly:-2, Asn:0,  Asp:0,  Glu:2,  Gln:5,  His:0,  Arg:1,  Lys:1,  Met:0,  Ile:-3, Leu:-2, Val:-2, Phe:-3, Tyr:-1, Trp:-2]
    [-3, -1, 0, -2, -2, 2, 1, 1, 0, 0, 8, 0, -1, -2, -3, -3, -2, -1, 2, -2],
    # H   His:[Cys:-3,Ser:-1, Thr:0,  Pro:-2, Ala:-2, Gly:-2, Asn:1,  Asp:1,  Glu:0,  Gln:0,  His:8,  Arg:0,  Lys:-1, Met:-2, Ile:-3, Leu:-3, Val:-2, Phe:-1, Tyr:2,  Trp:-2]
    [-3, -1, -1, -2, -1, -2, 0, -2, 0, 1, 0, 5, 2, -1, -3, -2, -3, -3, -2, -3],
    # R    Arg:[Cys:-3,Ser:-1, Thr:-1, Pro:-2, Ala:-1, Gly:-2, Asn:0,  Asp:-2, Glu:0,  Gln:1,  His:0,  Arg:5,  Lys:2,  Met:-1, Ile:-3, Leu:-2, Val:-3, Phe:-3, Tyr:-2, Trp:-3]
    [-3, 0, 0, -1, -1, -2, 0, -1, 1, 1, -1, 2, 5, -1, -3, -2, -3, -3, -2, -3],
    # K     Lys:[Cys:-3,Ser:0,  Thr:0,  Pro:-1, Ala:-1, Gly:-2, Asn:0,  Asp:-1, Glu:1,  Gln:1,  His:-1, Arg:2,  Lys:5,  Met:-1, Ile:-3, Leu:-2, Val:-3, Phe:-3, Tyr:-2, Trp:-3]
    [-1, -1, -1, -2, -1, -3, -2, -3, -2, 0, -2, -1, -1, 5, 1, 2, -2, 0, -1, -1],
    # M   Met:[Cys:-1,Ser:-1, Thr:-1, Pro:-2, Ala:-1, Gly:-3, Asn:-2, Asp:-3, Glu:-2, Gln:0,  His:-2, Arg:-1, Lys:-1, Met:5,  Ile:1,  Leu:2,  Val:-2, Phe:0,  Tyr:-1, Trp:-1]
    [-1, -2, -2, -3, -1, -4, -3, -3, -3, -3, -3, -3, -3, 1, 4, 2, 1, 0, -1, -3],
    # I   Ile:[Cys:-1,Ser:-2, Thr:-2, Pro:-3, Ala:-1, Gly:-4, Asn:-3, Asp:-3, Glu:-3, Gln:-3, His:-3, Arg:-3, Lys:-3, Met:1,  Ile:4,  Leu:2,  Val:1,  Phe:0,  Tyr:-1, Trp:-3]
    [-1, -2, -2, -3, -1, -4, -3, -4, -3, -2, -3, -2, -2, 2, 2, 4, 3, 0, -1, -2],
    # L   Leu:[Cys:-1,Ser:-2, Thr:-2, Pro:-3, Ala:-1, Gly:-4, Asn:-3, Asp:-4, Glu:-3, Gln:-2, His:-3, Arg:-2, Lys:-2, Met:2,  Ile:2,  Leu:4,  Val:3,  Phe:0,  Tyr:-1, Trp:-2]
    [-1, -2, -2, 2, 0, -3, 3, -3, -2, -2, 3, -3, -2, 1, 3, 1, 4, -1, -1, -3],
    # V  Val:[Cys:-1,Ser:-2, Thr:-2, Pro:-2, Ala:0,  Gly:-3, Asn:-3, Asp:-3, Glu:-2, Gln:-2, His:-3, Arg:-3, Lys:-2, Met:1,  Ile:3,  Leu:1,  Val:4,  Phe:-1, Tyr:-1, Trp:-3]
    [2, -2, -2, -4, -2, -3, -3, -3, -3, -3, -1, -3, -3, 0, 0, 0, -1, 6, 3, 1],
    # F     Phe:[Cys:-2,Ser:-2, Thr:-2, Pro:-4, Ala:-2, Gly:-3, Asn:-3, Asp:-3, Glu:-3, Gln:-3, His:-1, Arg:-3, Lys:-3, Met:0,  Ile:0,  Leu:0,  Val:-1, Phe:6,  Tyr:3,  Trp:1]
    [2, 2, 2, 3, -2, -3, -2, -3, -2, 1, 2, -2, -2, -1, -1, -1, -1, 3, 7, 2],
    # Y   Tyr:[Cys:-2,Ser:-2, Thr:-2, Pro:-3, Ala:-2, Gly:-3, Asn:-2, Asp:-3, Glu:-2, Gln:-1, His:2,  Arg:-2, Lys:-2, Met:-1, Ile:-1, Leu:-1, Val:-1, Phe:3,  Tyr:7,  Trp:2]
    [2, 3, 3, 4, -3, -2, 4, 4, 3, 2, -2, -3, -3, -1, -3, -2, -3, 1, 2, 11]
    # W     Trp:[Cys:-2,Ser:-3, Thr:-3, Pro:-4, Ala:-3, Gly:-2, Asn:-4, Asp:-4, Glu:-3, Gln:-2, His:-2, Arg:-3, Lys:-3, Met:-1, Ile:-3, Leu:-2, Val:-3, Phe:1,  Tyr:2,  Trp:11]
]


# map characters to blosum62 index
def get_Aminoacid_index(chr):
    if chr == 'C':
        return 0
    elif chr == 'S':
        return 1
    elif chr == 'T':
        return 2
    elif chr == 'P':
        return 3
    elif chr == 'A':
        return 4
    elif chr == 'G':
        return 5
    elif chr == 'N':
        return 6
    elif chr == 'D':
        return 7
    elif chr == 'E':
        return 8
    elif chr == 'Q':
        return 9
    elif chr == 'H':
        return 10
    elif chr == 'R':
        return 11
    elif chr == 'K':
        return 12
    elif chr == 'M':
        return 13
    elif chr == 'I':
        return 14
    elif chr == 'L':
        return 15
    elif chr == 'V':
        return 16
    elif chr == 'F':
        return 17
    elif chr == 'Y':
        return 18
    elif chr == 'W':
        return 19


# get two Amino acid score
def getScore(chr1, chr2):
    return blosum62[get_Aminoacid_index(chr1)][get_Aminoacid_index(chr2)]


# get the maxmum
def maxmum(*agrs):
    digits = []
    for value in agrs:
        digits.append(value)
    return [max(digits), digits.index(max(digits))]


# construct the H and D matrix
def construct_HD(*str):
    # H,D is a 2D matrix for two residue sequence
    if len(str) == 2:
        str1 = str[0]
        str2 = str[1]
        # H is score matrix
        H = np.zeros((len(str1) + 1, len(str2) + 1))
        # D is direction matrix and 0,1,2,3,4,5,6,7 represent Right, Top right,..., Bottom right, while 2, 3, 4 can appear
        D = np.zeros((len(str1) + 1, len(str2) + 1))
        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                score = getScore(str1[i - 1], str2[j - 1])
                if D[i - 1][j] == 2:
                    value_index = maxmum(0, H[i - 1][j] - repeat_cost, H[i - 1][j - 1] + score,
                                         H[i][j - 1] - first_cost)
                    H[i][j] = value_index[0]
                    if H[i][j] == (H[i - 1][j] - repeat_cost):
                        D[i][j] = 2
                    elif H[i][j] == (H[i - 1][j - 1] + score):
                        D[i][j] = 3
                    elif H[i][j] == (H[i][j - 1] - first_cost):
                        D[i][j] = 4
                elif D[i][j - 1] == 4:
                    value_index = maxmum(0, H[i - 1][j] - first_cost, H[i - 1][j - 1] + score,
                                         H[i][j - 1] - repeat_cost)
                    H[i][j] = value_index[0]
                    if H[i][j] == (H[i - 1][j] - first_cost):
                        D[i][j] = 2
                    elif H[i][j] == (H[i - 1][j - 1] + score):
                        D[i][j] = 3
                    elif H[i][j] == (H[i][j - 1] - repeat_cost):
                        D[i][j] = 4
                else:
                    value_index = maxmum(0, H[i - 1][j] - first_cost, H[i - 1][j - 1] + score, H[i][j - 1] - first_cost)
                    H[i][j] = value_index[0]
                    if H[i][j] == (H[i - 1][j] - first_cost):
                        D[i][j] = 2
                    elif H[i][j] == (H[i - 1][j - 1] + score):
                        D[i][j] = 3
                    elif H[i][j] == (H[i][j - 1] - first_cost):
                        D[i][j] = 4
        return [H, D]
    # H,D is a 3D cube for three residue sequence
    elif len(str) == 3:
        str1 = str[0]
        str2 = str[1]
        str3 = str[2]
        # H is score cube
        H = np.zeros((len(str1) + 1, len(str2) + 1, len(str3) + 1))
        # D is direction cube
        # noting that 1,2,3,4,5,6,7 represent a direction vector (0,0,1),(0,1,0),... all can appear except 0!
        D = np.zeros((len(str1) + 1, len(str2) + 1, len(str3) + 1))
        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                for k in range(1, len(str3) + 1):
                    #   score1
                    if D[i][j][k - 1] == 7 or D[i][j][k - 1] == 6:
                        score1 = (-2) * first_cost
                    elif D[i][j][k - 1] == 1:
                        score1 = (-2) * repeat_cost
                    else:
                        score1 = -(repeat_cost + first_cost)
                    #  score2
                    if D[i][j - 1][k] == 7 or D[i][j - 1][k - 1] == 5:
                        score2 = (-2) * first_cost
                    elif D[i][j - 1][k] == 2:
                        score2 = (-2) * repeat_cost
                    else:
                        score2 = -(repeat_cost + first_cost)
                    #  score3
                    if D[i][j - 1][k - 1] == 1 or D[i][j - 1][k - 1] == 2 or D[i][j - 1][k - 1] == 3:
                        score3 = getScore(str2[j - 1], str3[k - 1]) - repeat_cost
                    else:
                        score3 = getScore(str2[j - 1], str3[k - 1]) - first_cost
                    #  score4
                    if D[i - 1][j][k] == 4:
                        score4 = (-2) * repeat_cost
                    elif D[i - 1][j][k] == 7 or D[i - 1][j][k] == 3:
                        score4 = (-2) * first_cost
                    else:
                        score4 = -(repeat_cost + first_cost)
                    #   score5
                    if D[i - 1][j][k - 1] == 1 or D[i - 1][j][k - 1] == 4 or D[i - 1][j][k - 1] == 5:
                        score5 = getScore(str1[i - 1], str3[k - 1]) - repeat_cost
                    else:
                        score5 = score5 = getScore(str1[i - 1], str3[k - 1]) - first_cost
                    #   score6
                    if D[i - 1][j - 1][k] == 2 or D[i - 1][j - 1][k] == 4 or D[i - 1][j - 1][k] == 6:
                        score6 = getScore(str1[i - 1], str2[k - 1]) - repeat_cost
                    else:
                        score6 = getScore(str1[i - 1], str2[k - 1]) - first_cost
                    #   score7
                    score7 = getScore(str1[i - 1], str2[j - 1]) + getScore(str2[j - 1], str3[k - 1]) + getScore(
                        str1[i - 1], str3[k - 1])
                    value_index = maxmum(0, H[i][j][k - 1] + score1, H[i][j - 1][k] + score2,
                                         H[i][j - 1][k - 1] + score3, H[i - 1][j][k] + score4,
                                         H[i - 1][j][k - 1] + score5, H[i - 1][j - 1][k] + score6,
                                         H[i - 1][j - 1][k - 1] + score7)
                    H[i][j][k] = value_index[0]
                    D[i][j][k] = value_index[1]
        return [H, D]


# search the max value and its index
def MtrMaxValue(Mtr):
    maxvalue = 0
    line = 0
    column = 0
    for n in range(0, len(Mtr)):
        if max(Mtr[n]) > maxvalue:
            maxvalue = max(Mtr[n])
            line = n
        column = np.argmax(Mtr[line])
    return [maxvalue, line, column]


def CubeMaxValue(cube):
    maxvalue = 0
    i = 0
    j = 0
    k = 0
    for n in range(len(cube)):
        for m in range(len(cube[0])):
            if max(cube[n][m]) > maxvalue:
                maxvalue = max(cube[n][m])
                i = n
                j = m
    k = np.argmax(cube[i][j])
    return [maxvalue, i, j, k]


# traceback path
def traceback(H, D, *str):
    if len(str) == 2:
        max = MtrMaxValue(H)
        stop = max[1:3]
        string1 = ''
        string2 = ''
        str1 = str[0]
        str2 = str[1]
        for counts in range(100000):
            if H[max[1]][max[2]] == 0:
                break
            elif H[max[1]][max[2]] != 0:
                if D[max[1]][max[2]] == 2:
                    string1 += str1[max[1] - 1]
                    string2 += '-'
                    max[1] -= 1
                elif D[max[1]][max[2]] == 3:
                    string1 += str1[max[1] - 1]
                    string2 += str2[max[2] - 1]
                    max[1] -= 1
                    max[2] -= 1
                elif D[max[1]][max[2]] == 4:
                    string1 += '-'
                    string2 += str2[max[2]]
                    max[2] -= 1
        print('Similar Sequence:')
        # A
        print('A: ' + string1[::-1])
        print('Start ', end='')
        print(max[1] + 1, end='')
        print('     Stop ', end='')
        print(stop[0])
        # B
        print('B: ' + string2[::-1])
        print('Start ', end='')
        print(max[2] + 1, end='')
        print('     Stop ', end='')
        print(stop[1])
        # Total Score and condition
        print('Total Score: ', end='')
        print(max[0], end='')
        print('     Taking:      first_cost = ', end='')
        print(first_cost, end='')
        print('      repeat_cost = ', end='')
        print(repeat_cost)

    elif len(str) == 3:
        max = CubeMaxValue(H)
        stop = max[1:4]
        string1 = ''
        string2 = ''
        string3 = ''
        str1 = str[0]
        str2 = str[1]
        str3 = str[2]
        for counts in range(10000000000):
            if H[max[1]][max[2]][max[3]] == 0:
                break
            elif H[max[1]][max[2]][max[3]] != 0:
                if D[max[1]][max[2]][max[3]] == 1:
                    string1 += '-'
                    string2 += '-'
                    string3 += str3[max[3] - 1]
                    max[3] -= 1
                elif D[max[1]][max[2]][max[3]] == 2:
                    string1 += '-'
                    string2 += str2[max[2] - 1]
                    string3 += '-'
                    max[2] -= 1
                elif D[max[1]][max[2]][max[3]] == 3:
                    string1 += '-'
                    string2 += str2[max[2] - 1]
                    string3 += str3[max[3] - 1]
                    max[2] -= 1
                    max[3] -= 1
                elif D[max[1]][max[2]][max[3]] == 4:
                    string1 += str1[max[1] - 1]
                    string2 += '-'
                    string3 += '-'
                    max[1] -= 1
                elif D[max[1]][max[2]][max[3]] == 5:
                    string1 += str1[max[1] - 1]
                    string2 += '-'
                    string3 += str3[max[3] - 1]
                    max[1] -= 1
                    max[3] -= 1
                elif D[max[1]][max[2]][max[3]] == 6:
                    string1 += str1[max[1] - 1]
                    string2 += str2[max[2] - 1]
                    string3 += '-'
                    max[1] -= 1
                    max[2] -= 1
                elif D[max[1]][max[2]][max[3]] == 7:
                    string1 += str1[max[1] - 1]
                    string2 += str2[max[2] - 1]
                    string3 += str3[max[3] - 1]
                    max[1] -= 1
                    max[2] -= 1
                    max[3] -= 1
        print('Similar Sequence:')
        # A
        print('A: ' + string1[::-1])
        print('Start ', end='')
        print(max[1] + 1, end='')
        print('     Stop ', end='')
        print(stop[0])
        # B
        print('B: ' + string2[::-1])
        print('Start ', end='')
        print(max[2] + 1, end='')
        print('     Stop ', end='')
        print(stop[1])
        # C
        print('C: ' + string3[::-1])
        print('Start ', end='')
        print(max[3] + 1, end='')
        print('     Stop ', end='')
        print(stop[2])
        # Total Score and condition
        print('Total Score: ', end='')
        print(max[0], end='')
        print('     Taking:      first_cost = ', end='')
        print(first_cost, end='')
        print('      repeat_cost = ', end='')
        print(repeat_cost)


# Main
# Amino acid sequence
phkrs_lysRS = 'HWADYIADKIIRERGEKEKYVVESGITPSGYVHVGNFRELFTAYIVGHALRDKGYEVRHIHMWDDYDRFRKVPRNVPQEWKDYLGMPISEVPDPWGCHESYAEHFMRKFEEEVEKLGIEVDLLYASELYKRGEYSEEIRLAFEKRDKIMEILNKYREIAKQPPLPENWWPAMVYCPEHRREAEIIEWDGGWKVKYKCPEGHEGWVDIRSGNVKLRWRVDWPMRWSHFGVDFEPAGKDHLVAGSSYDTGKEIIKEVYGKEAPLSLMYEFVGIKGQNVILLSDLYEVLEPGLVRFIYARHRPNKEIKIDLGLGILNLYDEFEKVERIYFGVEGEELRRTYELSMPKKPERLVAQAPFRFLAVLVQLPHLTEEDIINVLIKQGHIPRDLSKEDVERVKLRINLARNWVKKYAPEDVKFSILEKPPEVEVSEDVREAMNEVAEWLENHEEFSVEEFNNILFEVAKRRGISSREWFSTLYRLFIGKERGPRLASFLASLDRSFVIKRLRLEGK'
Yeast_lysRS = 'ASTANMISQLKKLSIAEPAVAKDSHPDVNIVDLMRNYISQELSKISGVDSSLIFPALEWTNTMERGDLLIPIPRLRIKGANPKDLAVQWAEKFPCGDFLEKVEANGPFIQFFFNPQFLAKLVIPDILTRKEDYGSCKLVENKKVIIEFSSPNIAKPFHAGHLRSTIIGGFLANLYEKLGWEVIRMNYLGDWGKQFGLLAVGFERYGNEEALVKDPIHHLFDVYVRINKDIEEEGDSIPLEQSTNGKAREYFKRMEDGDEEALKIWKRFREFSIEKYIDTYARLNIKYDVYSGESQVSKESMLKAIDLFKEKGLTHEDKGAVLIDLTKFNKKLGKAIVQKSDGTTLYLTRDVGAAMDRYEKYHFDKMIYVIASQQDLHAAQFFEILKQMGFEWAKDLQHVNFGMVQGMSTRKGTVVFLDNILEETKEKMHEVMKKNENKYAQIEHPEEVADLVGISAVMIQDMQGKRINNYEFKWERMLSFEGDTGPYLQYAHSRLRSVERNASGITQEKWINADFSLLKEPAAKLLIRLLGQYPDVLRNAIKTHEPTTVVTYLFKLTHQVSSCYDVLWVAGQTEELATARLALYGAARQVLYNGMRLLGLTPVERM'
Thermophilus_gluRS = 'MVVTRIAPSPTGDPHVGTAYIALFNYAWARRNGGRFIVRIEDTDRARYVPGAEERILAALKWLGLSYDEGPDVGGPHGPYRQSERLPLYQKYAEELLKRGWAYRAFETPEELEQIRKEKGGYDGRARNIPPEEAEERARRGEPHVIRLKVPRPGTTEVKDELRGVVVYDNQEIPDVVLLKSDGYPTYHLANVVDDHLMGVTDVIRAEEWLVSTPIHVLLYRAFGWEAPRFYHMPLLRNPDKTKISKRKSHTSLDWYKAEGFLPEALRNYLCLMGFSMPDGREIFTLEEFIQAFTWERVSLGGPVFDLEKLRWMNGKYIREVLSLEEVAERVKPFLREAGLSWESEAYLRRAVELMRPRFDTLKEFPEKARYLFTEDYPVSEKAQRKLEEGLPLLKELYPRLRAQEEWTEAALEALLRGFAAEKGVKLGQVAQPLRAALTGSLETPGLFEILALLGKERALRRLERALA'
t1 = time()
HD = construct_HD(phkrs_lysRS, Yeast_lysRS, Thermophilus_gluRS)
t2 = time()
H = HD[0]
D = HD[1]
traceback(H, D, phkrs_lysRS, Yeast_lysRS, Thermophilus_gluRS)
t3 = time()
print('Time cost of construct HD:     ' + str(t2 - t1) + ' s')
print('Time cost of traceback:      ' + str(t3 - t2) + ' s')

import numpy as np


def get_line(p1: list, p2: list):
    # p1: [x, y] -> p1' = [x, y, 1]
    pp1 = [p1[0], p1[1], 1]
    pp2 = [p2[0], p2[1], 1]
    return np.cross(pp1, pp2)


# This function returns the vanishing line equation
def get_vanish_line(list1, list2):
    # list1: 4 points -> 2 line -> 1 vanish point
    # same as line2
    # Equation of lines
    l1 = get_line(list1[0], list1[1])
    l2 = get_line(list1[2], list1[3])
    p1 = get_line(list2[0], list2[1])
    p2 = get_line(list2[2], list2[3])

    temp1 = np.cross(l1, l2)
    temp2 = np.cross(p1, p2)
    # Vanishing Points
    v1 = temp1 / temp1[2]
    v2 = temp2 / temp2[2]
    return np.cross(v1, v2)


# This function returns the perpendicular lines required for metric rectification
def get_perpendicular_line(list1, list2):
    l1 = get_line(list1[0], list1[1])
    m1 = get_line(list1[2], list1[1])

    l2 = get_line(list2[0], list2[1])
    m2 = get_line(list2[2], list2[1])

    return l1 / l1[2], m1 / m1[2], l2 / l2[2], m2 / m2[2]


def get_symmetric_matrix(list1, list2):
    l1, m1, l2, m2 = get_perpendicular_line(list1, list2)
    C = np.array([-l1[1] * m1[1], -l2[1] * m2[1]])
    A = np.array([[l1[0] * m1[0], l1[0] * m1[1] + l1[1] * m1[0]], [l2[0] * m2[0], l2[0] * m2[1] + l2[1] * m2[0]]])
    s = np.matmul(np.linalg.inv(A), C)
    # symmetric matrix
    return np.array([[s[0], s[1]], [s[1], 1]])


# Returns the affine rectification homography matrix
def get_affine_homography_matrix(list1, list2, method="8p"):
    if method == "8p":
        line = get_vanish_line(list1, list2)
        return np.array([[1, 0, 0], [0, 1, 0], [line[0] / line[2], line[1] / line[2], 1]])
    elif method == "6p":
        s_mat = get_symmetric_matrix(list1, list2)
        U, D, V = np.linalg.svd(s_mat)
        D_sqrt = np.sqrt(D)
        Dm = np.array([[D_sqrt[0], 0], [0, D_sqrt[1]]])

        # The final A matrix
        Am = np.matmul(np.matmul(U, Dm), V)
        return np.array([[Am[0][0], Am[0][1], 0], [Am[1][0], Am[1][1], 0], [0, 0, 1]])

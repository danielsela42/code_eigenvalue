import numpy as np
import re

def get_gvects(g_file):
    ''' Reads g_file and gets the vectors as a list
    '''
    with open(g_file) as f:
        raw = f.read()

    exp_expr = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?" # re for data in exponential format
    #header_format_re = " *(?P<f1>{})[ \t]+(?P<f2>{})[ \t]+[ \t]*".format(exp_expr, exp_expr, exp_expr, exp_expr)
    header_format_re = " *(?P<f1>{})[ \t]*(?P<f2>{})[ \t]*".format(exp_expr, exp_expr)
    header_rec = re.compile(header_format_re)

    g_vects = list()
    for raw_line in raw.split('\n'):
        if raw_line:
            values = header_rec.match(raw_line)
            g_vects.append(np.array([float(values.group('f1')), float(values.group('f2'))]))
    return g_vects

def reorder(gvects1, gvects2,error):
    ''' Reorder g_vects1 to the order of gvects2 
    '''
    gvects = list()

    for vect in gvects2:
        count = 0
        for vect1 in gvects1:
            if abs(vect1[0] - vect[0]) < error and abs(vect1[1] - vect[1]) < error:
                print(vect1, vect)
                gvects.append(vect1)
                count += 1
        if count != 1: raise Exception("ERROR: More than one matching vector")
    
    return gvects

def construct_matrix(n_g, g_vects, neigh_distances, off_elem, factor, k, error):
    ''' construct Hermitian matrix

    parameters: n_g = # of vectors in g_vects
                g_vects = list of g vectors
                neigh_distances = list of near neighbor distance starting at nearest
                off_elem = off-diagonal element
                k - a k vector includes 0 and magnitudes of order 0.1
    
    returns matrix as numpy array
    '''
    matrix_list = list()
    for row in range(n_g):
            g1_row = list()
            print(np.linalg.norm(g_vects[row]))
            for col in range(n_g):
                g1 = g_vects[row]
                g2 = g_vects[col]
                if row == col:
                    # Kinetic term
                    elem = factor*(np.linalg.norm(np.add(k, g1)))**2
                else:
                    # Layer interaction and hopping
                    g2_g1_diff = np.subtract(g2, g1)
                    diff_mag = np.linalg.norm(g2_g1_diff)

                    neigh_distance = neigh_distances[0]
                    elem = complex(0., 0.)
                    if abs(diff_mag - neigh_distance) < error:
                        if row > col: elem = off_elem
                        else: elem = off_elem.conjugate()
                g1_row.append(elem)
            matrix_list.append(g1_row)
    return np.array(matrix_list)


def eigens(matrix):
    '''
    n_g = 127
    V = 33.5
    g_vects = get_gvects(gfile) # These are read from the file
    neigh_distances = [1]
    off_elem = V*np.exp(complex(0, phase))
    factor = -8.401847020709548
    k = np.array([0, 0])
    error = 10**(-8)

    matrix = construct_matrix(n_g, g_vects, neigh_distances, off_elem ,factor, k, error)
    '''

    # Calculate eigenvalues and vectors and perform checks
    eigvals_prev, eigvecs = np.linalg.eigh(matrix)
    eigvals = list()
    i=0
    for val in eigvals_prev:
        # Check that eigenvalues are real
        if not val.imag == 0:
            raise Exception("ERROR: Failed to calculated real energies")

        # Check that it is an eigenvalue:
        eigvec_mag = np.linalg.norm(np.subtract(matrix.dot(eigvecs[:, i]), val*eigvecs[:, i]))
        matrix_det = np.linalg.det(matrix) - val*n_g
        print("determinant: ", i, val, matrix_det, eigvec_mag)
        i += 1
    k_eigen_list = [(eigvals[i], eigvecs[:, i]) for i in range(n_g)]
    k_eigen_list.sort(key=lambda tup: tup[0])
    return k_eigen_list

if __name__=='__main__':
    gfile1 = "/home/dsela/ut_projects/cmp/code_eigenvalue/Gbasis_127_1.txt"
    gfile2 = "/home/dsela/ut_projects/cmp/code_eigenvalue/Gbasis_127_2.txt"

    phase = 120

    n_g = 127
    V = 33.5
    neigh_distances = [1]
    off_elem = V*np.exp(complex(0, phase*np.pi/180))
    factor = -8.401847020709548
    k = np.array([0, 0])
    error = 10**(-8)

    g_vects1_prev = get_gvects(gfile1)
    g_vects2 = get_gvects(gfile1)
    g_vects1 = reorder(g_vects1_prev, g_vects2, error)

    matrix1 = construct_matrix(n_g, g_vects1, neigh_distances, off_elem, factor, k, error)
    matrix2 = construct_matrix(n_g, g_vects1, neigh_distances, off_elem, factor, k, error)

    # Compare matrix component-wise
    for row in range(n_g):
        for col in range(n_g):
            print(matrix1[row, col], matrix2[row,col])
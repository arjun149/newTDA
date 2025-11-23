import gudhi
import numpy as np

def compute_persistent_homology(time_series):
    """
    Return betti_numbers: list of Betti numbers / features. 
    """
    rips_complex = gudhi.RipsComplex(points=time_series, max_edge_length=1.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    persistence = simplex_tree.persistence()
    
    # Extract Betti numbers for dimensions 0, 1, and 2
    betti_numbers = [len([b for b in persistence if b[0] == i]) for i in range(3)]
    
    return betti_numbers

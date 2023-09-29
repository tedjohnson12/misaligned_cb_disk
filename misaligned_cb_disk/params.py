import numpy as np

G = 1

def get_star_masses(Mb:float,fb:float):
    """
    Get the masses of the two stars given
    the total mass and the ratio between them.
    """
    M2 = Mb * fb
    M1 = Mb*(1-fb)
    return M1, M2

class Binary:
    def __init__(
        self,
        Mb:float,
        fb:float,
        ab:float,
        eb:float,
        Tb:float
    ):
        self.Mb = Mb
        self.fb = fb
        self.ab = ab
        self.eb = eb
        self.Tb = Tb
    
    

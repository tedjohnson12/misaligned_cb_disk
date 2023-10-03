
from rebound import Simulation, Particle

G = 1

def get_star_masses(mb:float,fb:float):
    """
    Get the masses of the two stars given
    the total mass and the ratio between them.
    """
    m2 = mb * fb
    m1 = mb*(1-fb)
    return m1, m2

class Binary:
    name1 = 'm1'
    name2 = 'm2'
    def __init__(
        self,
        mb:float,
        fb:float,
        ab:float,
        eb:float,
        Tb:float
    ):
        self.mb = mb
        self.fb = fb
        self.ab = ab
        self.eb = eb
        self.Tb = Tb
    @property
    def m1(self)->float:
        """
        The mass of Star 1

        :type: float
        """
        m1, _ = get_star_masses(self.mb,self.fb)
        return m1
    @property
    def m2(self)->float:
        """
        The mass of Star 2

        :type: float
        """
        _, m2 = get_star_masses(self.mb,self.fb)
        return m2
    def add_to_sim(self,sim:Simulation):
        star1 = Particle(
            simulation=sim,
            hash=self.name1,
            m=self.m1
        )
        sim.add(star1)
        star2 = Particle(
            simulation=sim,
            a=self.ab,
            e=self.eb,
            hash=self.name2,
            m=self.m2,
            primary=star1
        )
        sim.add(star2)
        sim.move_to_com()
    
    
    

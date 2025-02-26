

from numpy.random import binomial, normal


class proxyModel():
    def __init__(self, pZ, pXgivenZ, proxyGivenZ):
        """
        setting up linear Gaussian environment: 
        1. pZ is a Gaussain with pZ[0] be the mean and pZ[1] be the standard deviation
        2. pXgivenZ is a binomial conditioanl with pXgivenZ[0] be the dist when Z=0  and pXgivenZ[1] be the dis when Z = 1
        3. proxyGivenZ is the coefficents and the intercept of the linear function that generates the proxy

        """
        

        self.pZ = pZ                  
        self.pU = 0.8
        self.pXgivenZ = pXgivenZ    
        self.proxyGivenZ = proxyGivenZ  

        self.action_space = 3        #0:do(x=0), 1:do(x=1) and 2:do() 
        self.observation_space = 1   #number of context can be observed
        
        self.z = 0             #context of current environment
        self.proxy =  proxyGivenZ[0] * self.z + proxyGivenZ[1] + normal(size=5)  
        self.x = 0


       
    
    def sample_action(self):   
        
        if self.z <= 0:
            p = self.pXgivenZ[0]
        else:
            p = self.pXgivenZ[1]      
        x = binomial(1, p)
        return x
    

    def sample_reward(self, z, x):
        
        if (z >= 5 and  x== 0) or (z < 5 and x == 1):
            y = 1
        else:
            y = 0 
        return y
    
    def sample_proxy(self, z):

        a = self.proxyGivenZ[0]     
        b = self.proxyGivenZ[1]   

        # linear Gaussian transformation
        proxy = a * self.z + b + normal(size=5)
        

        assert proxy.shape == (1,5)
        return proxy

    def sample_z(self):
        z = normal(loc=self.pZ[0], scale=self.pZ[1], size=1)
        return z


    def pull(self, action):
        if action == 2:                                        #oberservation mode

            if self.z == 0:   
                #generate passive x                           
                x = binomial(1, self.pXgivenZ[0])              
            else:
                #generate passive x
                x = binomial(1, self.pXgivenZ[1])  

            reward = self.sample_reward(self.z, x)       

        else:     
            reward = self.sample_reward(self.z, action)  


        #update the z 
        self.z = self.sample_z()
        
        #generate proxy and passive action
        self.proxy = self.sample_proxy(self.z)
        self.x = self.sample_action()
        
        return reward
    

        

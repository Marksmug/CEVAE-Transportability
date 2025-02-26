import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
Line2D._us_dashSeq    = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
Legend._ncol = property(lambda self: self._ncols)

def plot_totalReward(rewards, labels, pZ_source, pZ_target, prob_flag, causal_optimal):
    
    if prob_flag:
        proxy_model = 'Prob'
    else:
        proxy_model = 'Lin'    

    for i in range(len(rewards)):
        plt.plot(rewards[i], label = labels[i])
        
    plt.axhline(y=5000.0, color='b', linestyle='-.', label='optimal')
    #plt.axhline(y=causal_optimal*1000, color='r', linestyle='-.', label='Causal-optimal')
    plt.title('source=(%.0f' %pZ_source[0] + ', %.0f'%pZ_source[1] + ')\ntarget=(%.0f' %pZ_target[0] + ', %.0f'%pZ_target[1]+')')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.legend(loc= 'lower right')
    #plt.ylim(300, 1050)
    plt.savefig('results/totalReward('+proxy_model+')_['+str(pZ_target[0])+','+str(pZ_target[1])+']_withSource['+str(pZ_source[0])+','+str(pZ_source[1])+'].png', dpi = 400)

    plt.show()
    

def plot_stepReward(mean, std, labels, pZ_source, pZ_target,optimal = False):
    
    x = np.arange(0,len(mean[0]),1)
    for i in range(len(mean)):
        plt.plot(mean[i], label = labels[i])
        plt.fill_between(x, mean[i]-std[i], mean[i]+std[i], alpha=0.2)

        
    if optimal:
        plt.plot(x, 1*x, color='r', linestyle='-.', label = 'optimal')
    plt.xlabel("step")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    #plt.ylim(0, 1000)
    plt.show()
    
def plot_stepRegret(name, mean, std, labels, pZ_source, pZ_target):

    x = np.arange(0,len(mean[0]),1)
    colors = ['green', 'green', 'blue']
    linestyle =['--','-','-']
    agent_selection = [0,1,2]
    plt.figure(figsize=(8,5))
    for i in agent_selection:
        plt.plot(mean[i], label = labels[i], color = colors[i], linestyle=linestyle[i])
        plt.fill_between(x, mean[i]-std[i], mean[i]+std[i], alpha=0.2, color=colors[i])
    plt.title('source=(%.0f' %pZ_source[0] + ', %.0f'%pZ_source[1] + ')\ntarget=(%.0f' %pZ_target[0] + ', %.0f'%pZ_target[1]+')')

 
    plt.xlabel("step")
    plt.ylabel("Regret")
    plt.legend()
    plt.ylim(0, 200)
     # Adjust the layout
    plt.tight_layout()
    #plt.savefig('results/stepRegret('+proxy_model+')_['+str(pZ_target[0])+','+str(pZ_target[1])+']_withSource['+str(pZ_source[0])+','+str(pZ_source[1])+'].png', dpi=400)

    #plt.show()
    tikzplotlib.save('results/'+name)

def plot_stepPZ(mean, std, pZ_source, pZ_target):
    
    x = np.arange(0,len(mean),1)

    plt.plot(mean)
    plt.fill_between(x, mean-std, mean+std, alpha=0.2)
    plt.title('source=%.2f' %pZ_source + '\ntarget=%.2f' %pZ_target)
    plt.xlabel("step")
    plt.ylabel("P(z)")
    plt.legend()
    plt.xlim(0, 500)
    plt.savefig('stepPZ3')
    plt.show()
    
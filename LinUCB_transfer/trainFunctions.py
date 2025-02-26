import numpy as np
import algorithms as a
import torch.optim as optim

def train_source(env, agent, steps):
    """
    Training function for source domain
    """
    rewards = np.zeros(steps)
    for i in range(steps):
        observation = env.proxy
        # give causal agent the true context 
        if isinstance(agent, a.CausalAgent): 
            context = env.context
        else:
            context = observation
        
        action = agent.train_act(context)

        # following the passive action
        if action == 2:
            action = env.x
        
        reward = env.pull(action)
        rewards[i] = rewards[i-1] + reward 

        if isinstance(agent, a.CausalAgent):
            agent.train_learn(action, reward, context, observation)
        else:
            agent.train_learn(action, reward, context)

    return rewards



def train_stepReward(env, nEpisodes, steps, agent, optimal):
    """
    trainning function for the target domain
    """
    rewards_episode = []
    regret_episode = []
    target_Parameters_episode = []

    
    for i in range(nEpisodes):
        print("Episode: ", i+1)
        rewards = np.zeros(steps)
        regret = np.zeros(steps)

        for t in range(steps):
            #observed_action = env.sample_action()
            observation = env.proxy
           
            if isinstance(agent, a.CausalAgent):
                obs = observation.flatten()
                agent.memory.push(obs)

            action = agent.act(observation)
            reward = env.pull(action)
            rewards[t] = rewards[t-1] + reward                  
            regret[t] = optimal * t - rewards[t]              

            if isinstance(agent, a.CausalAgent):
                if (t+1)%agent.frequency == 0:
                    loss = agent.learn()
            else:
                agent.learn(action, reward, observation)

        rewards_episode.append(rewards)
        regret_episode.append(regret)

        # reset the agent every episode    
        agent.reset()
    
    # step reward 
    step_mean = np.mean(rewards_episode, axis = 0)
    step_std = np.std(rewards_episode, axis = 0)
  
    # step regret
    regret_mean = np.mean(regret_episode, axis = 0)
    regret_std = np.std(regret_episode, axis = 0)

    # total reward of each episode
    total_rewards = [rewards[steps-1] for rewards in rewards_episode ]                   


    
    return step_mean, step_std, regret_mean, regret_std, total_rewards
    
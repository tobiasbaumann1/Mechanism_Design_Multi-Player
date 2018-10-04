import matplotlib.pyplot as plt
import numpy as np
import math
import os
import logging
logging.basicConfig(filename='main.log',level=logging.DEBUG,filemode='w')
from Environments import Matrix_Game, Linear_Schelling_Game
from Agents import Actor_Critic_Agent, Critic_Variant, Simple_Agent
from Planning_Agent import Planning_Agent
import itertools

N_PLAYERS = 10
N_UNITS = 10 #number of nodes in the intermediate layer of the NN
MAX_REWARD_STRENGTH = 3
COST_PARAM = 0.0002
N_EPISODES = 2000

def run_game(N_EPISODES, env, players, action_flip_prob, planning_agent = None, revenue_neutral = True, 
    turn_off_after_ep_nr = math.inf):
    env.reset_all()
    avg_planning_rewards_per_round = []
    for episode in range(N_EPISODES):
        # initial observation
        s = env.reset()
        flag = isinstance(s, list)

        cum_planning_rs = [0]*len(players)
        while True:
            # choose action based on s
            if flag:
                actions = [player.choose_action(s[idx]) for idx, player in enumerate(players)]
            else:
                actions = [player.choose_action(s) for player in players]
            

            # take action and get next s and reward
            s_, rewards, done = env.step(actions)

            perturbed_actions = [(1-a if np.random.binomial(1,action_flip_prob) else a) for a in actions]

            if planning_agent is not None and episode < turn_off_after_ep_nr:
                planning_rs = planning_agent.choose_action(s,perturbed_actions)
                if revenue_neutral:
                    sum_planning_r = sum(planning_rs)
                    mean_planning_r = sum_planning_r / N_PLAYERS
                    planning_rs = [r-mean_planning_r for r in planning_rs]
                rewards = [ sum(r) for r in zip(rewards,planning_rs)]
                cum_planning_rs = [sum(r) for r in zip(cum_planning_rs, planning_rs)]
                # Training planning agent
                planning_agent.learn(s,perturbed_actions)
            logging.info('Actions:' + str(actions))
            logging.info('State after:' + str(s_))
            logging.info('Rewards: ' + str(rewards))
            logging.info('Done:' + str(done))

            for idx, player in enumerate(players):
                if flag:
                    player.learn(s[idx], actions[idx], rewards[idx], s_[idx], s, s_)
                else:
                    player.learn(s, actions[idx], rewards[idx], s_)

            # swap s
            s = s_

            # break while loop when done
            if done:
                for player in players:
                    player.learn_at_episode_end() 
                break
        avg_planning_rewards_per_round.append([r / env.step_ctr for r in cum_planning_rs])

        # status updates
        if (episode+1) % 100 == 0:
            print('Episode {} finished.'.format(episode + 1))
    return env.get_avg_rewards_per_round(), np.asarray(avg_planning_rewards_per_round)

def plot_results(data, legend, path, title, ylabel = 'Reward', exp_factor = 1):
    plt.figure()
    if np.ndim(data) > 1:
        for idx in range(data.shape[1]):
            avg_list = []
            avg = data[0,idx]
            for r in data[:,idx]:
                avg = exp_factor * r + (1-exp_factor) * avg
                avg_list.append(avg)
            first_idx = int(1 / exp_factor)
            plt.plot(range(first_idx,len(avg_list)),avg_list[first_idx:])
    else:
        avg_list = []
        avg = data[0]
        for r in data:
            avg = exp_factor * r + (1-exp_factor) * avg
            avg_list.append(avg)
        first_idx = int(1 / exp_factor)
        plt.plot(range(first_idx,len(avg_list)),avg_list[first_idx:])
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(legend)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path+'/' + title)
    plt.close()
    #plt.show()

def create_population(env,n_agents, use_simple_agents = False):    
    critic_variant = Critic_Variant.CENTRALIZED
    if use_simple_agents:
        l = [Simple_Agent(env, 
                      learning_rate=0.01,
                      gamma=0.9,
                      agent_idx = i,
                      critic_variant = critic_variant) for i in range(n_agents)]
    else:
        l = [Actor_Critic_Agent(env, 
                      learning_rate=0.01,
                      gamma=0.9,
                      n_units_actor = N_UNITS,
                      agent_idx = i,
                      critic_variant = critic_variant) for i in range(n_agents)]
    #Pass list of agents for centralized critic
    if critic_variant is Critic_Variant.CENTRALIZED:
        for agent in l:
            agent.pass_agent_list(l)
    return l

def run_game_and_plot_results(N_EPISODES, env,agents, planning_agent,
    revenue_neutral = False, max_reward_strength = None, cost_param = 0, value_fn_variant = 'exact', 
    symmetric = 'True', action_flip_prob = 0, 
    turn_off_after_ep_nr = math.inf):
    avg_rewards_per_round,avg_planning_rewards_per_round = run_game(N_EPISODES,env,agents,action_flip_prob,
        planning_agent = planning_agent, revenue_neutral = revenue_neutral, turn_off_after_ep_nr = turn_off_after_ep_nr)
    path = './Results/' + env.__str__() 
    if planning_agent is not None:
        path +='/' + ('revenue_neutral' if revenue_neutral else 'not_revenue_neutral')
        path += '/' + 'max_reward_strength_' + (str(MAX_REWARD_STRENGTH) if MAX_REWARD_STRENGTH is not None else 'inf')
        path += '/' + 'cost_parameter_' + str(COST_PARAM)
        path += '/' + value_fn_variant + '_value_function'
        path += '/' + 'symmetric' if symmetric else 'not_symmetric'
        if turn_off_after_ep_nr < math.inf:
            path += '/' + 'turning_off' 
        if action_flip_prob > 0:
            path += '/' + 'action_flip_prob'  + str(action_flip_prob)
    else:
        path += '/no_mechanism_design'

    plot_results(avg_rewards_per_round,[str(agent) for agent in agents],path,'Average Rewards', exp_factor=0.05)
    actor_a_prob_each_round = np.transpose(np.array([agent.log for agent in agents]))
    plot_results(actor_a_prob_each_round,[str(agent) for agent in agents],path,'Player Action Probabilities', ylabel = 'P(Cooperation)')
    avg_C_fraction = np.mean(actor_a_prob_each_round, axis = 1)
    plot_results(avg_C_fraction,['C_fraction'],path,'Fraction of cooperators', ylabel = '')
    social_welfare = np.array([env.calculate_social_welfare(C_frac) for C_frac in avg_C_fraction])
    plot_results(social_welfare,['Social welfare'],path,'Social welfare per round', ylabel = '')
    if planning_agent is not None:
        plot_results(avg_planning_rewards_per_round,[str(agent) for agent in agents],path,'Planning Rewards', exp_factor=0.05)
        cum_planning_rewards = np.sum(np.cumsum(np.absolute(avg_planning_rewards_per_round), axis = 0),axis = 1)
        plot_results(cum_planning_rewards,None,path,'Cumulative Additional Rewards', exp_factor=0.05)
        planning_a_prob_each_round = np.array(planning_agent.get_log())
        fear_and_greed_each_round = calc_fear_and_greed(planning_a_prob_each_round, env.FEAR, env.GREED)
        plot_results(planning_a_prob_each_round,
            ['D', 'C'] if symmetric else ['(D,D)', '(D,C)', '(C,D)', '(C,C)'],path,'Additional Rewards', ylabel = 'a_p')
        plot_results(fear_and_greed_each_round,['Fear', 'Greed'],path,'Modified Fear and Greed', ylabel = 'Fear/Greed')    
    final_C_fraction = avg_C_fraction[-1]
    final_R = env.calculate_social_welfare(final_C_fraction)
    avg_planning_rewards = cum_planning_rewards[-1] / N_EPISODES if planning_agent is not None else 0
    return final_C_fraction, final_R, avg_planning_rewards


def calc_fear_and_greed(data, base_fear, base_greed):
    assert(data.shape[1] == 2)
    if data.shape[2] == 2:
        fear = data[:,0,0]-data[:,1,0] + base_fear
        greed = data[:,0,1]-data[:,1,1] + base_greed
    else:
        fear = data[:,0]-data[:,1] + base_fear
        greed = data[:,0]-data[:,1] + base_greed
    return np.stack([fear,greed],axis = 1)

def run_main(args, mode, n_runs):
    d = {}
    d['args'] = args
    d['mode'] = mode
    print('Run game with arguments: ' + str(args))
    print('Mode: ' + str(mode))
    env = Linear_Schelling_Game(N_PLAYERS = N_PLAYERS, FEAR = args[3][0], GREED = args[3][1])
    print(env)
    agents = create_population(env,N_PLAYERS, use_simple_agents = True)
    if not (mode == 'No_mechanism_design'):
        planning_agent = Planning_Agent(env,agents,max_reward_strength = MAX_REWARD_STRENGTH, 
            cost_param = COST_PARAM, revenue_neutral = args[0],
            value_fn_variant = args[1], symmetric = args[2])
    final_C_fraction_list, final_R_list, avg_planning_rewards_list = [], [], []
    for i in range(n_runs):
        successful = False
        while not successful:
            successful = True
            if mode == 'No_mechanism_design':
                final_C_fraction, final_R, avg_planning_rewards = run_game_and_plot_results(N_EPISODES,env,agents,planning_agent = None) 
            else:
                if mode == 'With_mechanism_design':
                    final_C_fraction, final_R, avg_planning_rewards = run_game_and_plot_results(N_EPISODES,env,agents,
                            planning_agent = planning_agent,max_reward_strength = MAX_REWARD_STRENGTH, 
                            cost_param = COST_PARAM, revenue_neutral = args[0],
                            value_fn_variant = args[1], symmetric = args[2], action_flip_prob = 0)   
                else: 
                    if mode == 'Turning_off':
                        final_C_fraction, final_R, avg_planning_rewards = run_game_and_plot_results(2*N_EPISODES,env,agents,
                            planning_agent = planning_agent,max_reward_strength = MAX_REWARD_STRENGTH, 
                            cost_param = COST_PARAM, revenue_neutral = args[0],
                            value_fn_variant = args[1], symmetric = args[2], turn_off_after_ep_nr = N_EPISODES
                            , action_flip_prob = 0)
                planning_agent.reset()
            for agent in agents:
                agent.reset()
            if np.isnan(final_C_fraction):
                successful = False

        final_C_fraction_list.append(final_C_fraction)
        final_R_list.append(final_R)
        avg_planning_rewards_list.append(avg_planning_rewards)
    final_C_fraction_list, final_R_list, avg_planning_rewards_list = \
        np.array(final_C_fraction_list), np.array(final_R_list), np.array(avg_planning_rewards_list)

    # collect statistics for each: p(coop) at end, V, extra rewards per round, fear + greed at end, ...
    d['Final number of cooperators'], d['Final sum of rewards'],\
            d['Average extra rewards per round'] = np.mean(final_C_fraction_list), \
            np.mean(final_R_list), np.mean(avg_planning_rewards_list)
    d['Final number of cooperators - stdev'], d['Final sum of rewards - stdev'],\
            d['Average extra rewards per round - stdev'] = np.std(final_C_fraction_list), \
            np.std(final_R_list), np.std(avg_planning_rewards_list)
    return d

    
if __name__ == "__main__":
    # fear = [1,-1,1]
    # greed = [1,0.5,-1]
    # revenue_neutral=[False, True]
    # n_runs = 10

    fear = [1,-1,1]
    greed = [1,0.5,-1]
    revenue_neutral=[False]
    symmetric = [True]
    n_runs = 1
    
    value_fn_variant = ['exact']
    results = []
    for args in itertools.product(revenue_neutral,value_fn_variant,symmetric,zip(fear,greed)):
        results.append(run_main(args,'With_mechanism_design', n_runs))
        results.append(run_main(args,'No_mechanism_design', n_runs))
        results.append(run_main(args,'Turning_off', n_runs))
    
    with open('./Results/Summary.txt', 'w') as output_file:
        for d in results:
            for k,v in d.items():
                output_file.write(str(k) + ' >>> '+ str(v) + '\n\n')
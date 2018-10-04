import numpy as np

class Environment(object):
    def __init__(self, N_ACTIONS, N_PLAYERS, EPISODE_LENGTH, N_FEATURES = 0):
        self.N_ACTIONS = N_ACTIONS
        self.N_PLAYERS = N_PLAYERS
        self.N_FEATURES = N_FEATURES
        self.EPISODE_LENGTH = EPISODE_LENGTH
        self.step_ctr = 0
        self.ep_ctr = 0
        self.actions_list = []
        self.avg_rewards_per_round = []        
        self.reset()

    def step(self, actions):
        self.actions_list.append(actions)
        rewards = self.calculate_payoffs(actions)
        self.stored_rewards[:,self.step_ctr] = rewards
        self.update_state(actions)
        self.step_ctr += 1
        return self.state_to_observation(), rewards, self.is_done()

    def reset(self):
        self.s = self.initial_state()
        self.actions_list = []
        self.step_ctr = 0
        self.stored_rewards = np.zeros((self.N_PLAYERS,self.EPISODE_LENGTH))
        self.ep_ctr += 1
        return self.state_to_observation()

    def reset_all(self):
        self.actions_list = []
        self.avg_rewards_per_round = []
        self.ep_ctr = 0     

    def state_to_observation(self):
        return self.s

    def update_state(self, actions):
        pass

    def initial_state(self):
        return None

    def is_done(self):
        if self.step_ctr >= self.EPISODE_LENGTH:
            self.avg_rewards_per_round.append(np.mean(self.stored_rewards,axis=1))
            return True
        else:
            return False

    def get_avg_rewards_per_round(self):
        return np.asarray(self.avg_rewards_per_round)

class Linear_Schelling_Game(Environment):
    def __init__(self, N_PLAYERS,
        FEAR, GREED, EPISODE_LENGTH = 1, PAYOFF_ALL_C = 3, PAYOFF_ALL_D = 1):
        self.FEAR = FEAR
        self.GREED = GREED
        #Verify that it's a social dilemma
        assert(PAYOFF_ALL_C > PAYOFF_ALL_D)
        assert((FEAR > 0) or (GREED > 0))
        self.PAYOFF_C_IF_ALL_C = PAYOFF_ALL_C
        self.PAYOFF_C_IF_ALL_D = PAYOFF_ALL_D - FEAR
        self.PAYOFF_D_IF_ALL_C = PAYOFF_ALL_C + GREED
        self.PAYOFF_D_IF_ALL_D = PAYOFF_ALL_D
        super().__init__(N_ACTIONS = 2, N_PLAYERS = N_PLAYERS, EPISODE_LENGTH = EPISODE_LENGTH, N_FEATURES = 1)

    def initial_state(self):
        return np.zeros(1) #dummy feature 

    def calculate_payoffs(self, actions):
        assert(all([a == 0 or a == 1 for a in actions]))
        C_count = sum(actions) #number of cooperators
        C_payoff, D_payoff = self.calculate_C_and_D_payoffs(C_count)
        return [C_payoff if a == 1 else D_payoff for a in actions]

    def calculate_social_welfare(self,C_fraction):
        C_count = C_fraction * self.N_PLAYERS
        C_payoff, D_payoff = self.calculate_C_and_D_payoffs(C_count)
        return self.N_PLAYERS * (C_fraction*C_payoff+(1-C_fraction)*D_payoff)

    def calculate_C_and_D_payoffs(self,C_count):
        C_payoff = (C_count-1) / (self.N_PLAYERS-1) * self.PAYOFF_C_IF_ALL_C + \
            (1 - (C_count-1) / (self.N_PLAYERS-1)) * self.PAYOFF_C_IF_ALL_D
        D_payoff = C_count / (self.N_PLAYERS-1) * self.PAYOFF_D_IF_ALL_C + \
            (1 - C_count / (self.N_PLAYERS-1)) * self.PAYOFF_D_IF_ALL_D
        return C_payoff,D_payoff

    def __str__(self):
       return "Linear_Schelling_Game, Greed=" + str(self.GREED) + "_Fear=" + str(self.FEAR)

class Matrix_Game(Linear_Schelling_Game):
    def __init__(self, FEAR, GREED):
        super().__init__(N_PLAYERS = 2, FEAR = FEAR, GREED = GREED)

    def __str__(self):
        description = "Matrix_Game_Greed=" + str(self.GREED) + "_Fear=" + str(self.FEAR)
        return description
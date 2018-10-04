import tensorflow as tf
import numpy as np
import logging
logging.basicConfig(filename='Planning_Agent.log',level=logging.DEBUG,filemode='w')
from Agents import Agent

RANDOM_SEED = 5
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

class Planning_Agent(Agent):
    def __init__(self, env, underlying_agents, learning_rate=0.01,
        gamma = 0.95, max_reward_strength = None, cost_param = 0, revenue_neutral = False, 
        value_fn_variant = 'exact', symmetric = True):
        super().__init__(env, learning_rate, gamma)     
        self.underlying_agents = underlying_agents
        self.log = []
        self.max_reward_strength = max_reward_strength
        n_players = len(underlying_agents)
        self.revenue_neutral = revenue_neutral
        self.value_fn_variant = value_fn_variant
        self.symmetric = symmetric

        self.s = tf.placeholder(tf.float32, [1, env.N_FEATURES], "state")  
        self.a_player = tf.placeholder(tf.float32, [1, 1 if symmetric else n_players], "player_action")
        if value_fn_variant == 'exact':
            self.p_player = tf.placeholder(tf.float32, [1, 1 if symmetric else n_players], "player_action_prob")
            if symmetric:
                self.p_opp = tf.placeholder(tf.float32, [1, 1], "player_opp_action_prob")
        self.r_players = tf.placeholder(tf.float32, [1, n_players], "player_rewards")
        self.inputs = tf.concat([self.s,self.a_player],1)

        with tf.variable_scope('Policy_p', reuse = tf.AUTO_REUSE):
            self.l1 = tf.layers.dense(
                inputs=self.inputs,
                units=1 if symmetric else n_players,    # 1 output per agent
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0, .1),  # weights
                bias_initializer=tf.random_normal_initializer(0, .1),  # biases
                name='actions_planning'
            )

            if max_reward_strength is None:
                self.action_layer = self.l1
            else:
                self.action_layer = tf.sigmoid(self.l1)

        with tf.variable_scope('Vp'):
            if max_reward_strength is not None:
                self.vp = 2 * max_reward_strength * (self.action_layer - 0.5)
            else:
                self.vp = self.action_layer

        with tf.variable_scope('V_total'):
            if value_fn_variant == 'estimated':
                self.v = tf.reduce_sum(self.r_players) - env.T - env.S + 0.1
        with tf.variable_scope('cost_function'):
            if value_fn_variant == 'estimated':
                self.g_log_pi = tf.placeholder(tf.float32, [1, n_players], "player_gradients")
            if symmetric:
                assert(value_fn_variant == 'exact')
                self.g_p = self.p_player[0,0] * (1-self.p_player[0,0])
                self.g_Vp = self.g_p * tf.gradients(ys = self.vp[0,0],xs = self.a_player)[0][0,0]
                self.g_V = self.g_p * (self.p_opp * (2 * env.PAYOFF_C_IF_ALL_C - env.PAYOFF_D_IF_ALL_C - env.PAYOFF_C_IF_ALL_D) 
                    + (1-self.p_opp) * (env.PAYOFF_D_IF_ALL_C + env.PAYOFF_C_IF_ALL_D - 2 * env.PAYOFF_D_IF_ALL_D))
                self.loss = - underlying_agents[0].learning_rate * self.g_Vp * self.g_V
            else:                
                cost_list = []
                for underlying_agent in underlying_agents:
                    idx = underlying_agent.agent_idx
                    if value_fn_variant == 'estimated':
                        # policy gradient theorem
                        self.g_Vp = self.g_log_pi[0,idx] * self.vp[0,idx]
                        self.g_V = self.g_log_pi[0,idx] * (self.v[0,idx] if value_fn_variant == 'proxy' else self.v)
                    if value_fn_variant == 'exact':
                        self.g_p = self.p_player[0,idx] * (1-self.p_player[0,idx])
                        self.p_opp = self.p_player[0,1-idx]
                        self.g_Vp = self.g_p * tf.gradients(ys = self.vp[0,idx],xs = self.a_player)[0][0,idx]
                        self.g_V = self.g_p * (self.p_opp * (2 * env.R - env.T - env.S) 
                            + (1-self.p_opp) * (env.T + env.S - 2 * env.P))

                    #cost_list.append(- underlying_agent.learning_rate * tf.tensordot(self.g_Vp,self.g_V,1))
                    cost_list.append(- underlying_agent.learning_rate * self.g_Vp * self.g_V)

                if revenue_neutral:
                    extra_loss = cost_param * tf.norm(self.vp-tf.reduce_mean(self.vp))
                else:
                    extra_loss = cost_param * tf.norm(self.vp)
                self.loss = tf.reduce_sum(tf.stack(cost_list)) + extra_loss

        with tf.variable_scope('trainPlanningAgent', reuse = tf.AUTO_REUSE):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, 
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Policy_p'))  

        self.sess.run(tf.global_variables_initializer())

    def learn(self, s, a_player):
        s = s[np.newaxis,:]
        r_players = np.asarray(self.env.calculate_payoffs(a_player))
        if self.symmetric:
            for idx,a in enumerate(a_player):
                feed_dict = {self.s: s, self.a_player: np.asarray(a).reshape(1,1), 
                        self.r_players: r_players[np.newaxis,:]}
                if self.value_fn_variant == 'estimated':
                    g_log_pi = self.underlying_agents[idx].calc_g_log_pi(s,a)
                    feed_dict[self.g_log_pi] = g_log_pi
                if self.value_fn_variant == 'exact':
                    p_player = self.underlying_agents[idx].calc_action_probs(s)[0,-1]
                    feed_dict[self.p_player] = np.asarray(p_player).reshape(1,1)
                    p_opp = self.underlying_agents[1-idx].calc_action_probs(s)[0,-1]
                    feed_dict[self.p_opp] = np.asarray(p_opp).reshape(1,1)
        else:
            a_player = np.asarray(a_player)
            feed_dict = {self.s: s, self.a_player: a_player[np.newaxis,:], 
                        self.r_players: r_players[np.newaxis,:]}
            if self.value_fn_variant == 'estimated':
                g_log_pi_list = []
                for underlying_agent in self.underlying_agents:
                    idx = underlying_agent.agent_idx
                    g_log_pi_list.append(underlying_agent.calc_g_log_pi(s,a_player[idx]))
                g_log_pi_arr = np.reshape(np.asarray(g_log_pi_list),[1,-1])
                feed_dict[self.g_log_pi] = g_log_pi_arr
            if self.value_fn_variant == 'exact':
                p_player_list = []
                for underlying_agent in self.underlying_agents:
                    idx = underlying_agent.agent_idx
                    p_player_list.append(underlying_agent.calc_action_probs(s)[0,-1])
                p_player_arr = np.reshape(np.asarray(p_player_list),[1,-1])
                feed_dict[self.p_player] = p_player_arr
        self.sess.run([self.train_op], feed_dict)

        action,loss,g_Vp,g_V = self.sess.run([self.action_layer,self.loss,
            self.g_Vp,self.g_V], feed_dict)
        logging.info('Learning step')
        logging.info('Planning_action: ' + str(action))
        if self.value_fn_variant == 'estimated':
            vp,v = self.sess.run([self.vp,self.v],feed_dict)
            logging.info('Vp: ' + str(vp))
            logging.info('V: ' + str(v))
        logging.info('Gradient of V_p: ' + str(g_Vp))
        logging.info('Gradient of V: ' + str(g_V))
        logging.info('Loss: ' + str(loss))

    def get_log(self):
        return self.log

    def choose_action(self, s, a_player):
        logging.info('Player actions: ' + str(a_player))
        s = s[np.newaxis, :]
        if self.symmetric:
            a_plan = [self.sess.run(self.action_layer, {self.s: s, self.a_player: np.asarray(a).reshape(1,1)}) \
                    for a in a_player]
            a_plan = np.asarray(a_plan).reshape(-1)
        else:
            a_player = np.asarray(a_player)        
            a_plan = self.sess.run(self.action_layer, {self.s: s, self.a_player: a_player[np.newaxis,:]})[0,:]
        if self.max_reward_strength is not None:
            a_plan = 2 * self.max_reward_strength * (a_plan - 0.5)
        logging.info('Planning action: ' + str(a_plan))
        self.log.append(self.calc_conditional_planning_actions(s))
        return a_plan

    def calc_conditional_planning_actions(self,s):
        if self.symmetric:
            a_plan_D = self.sess.run(self.action_layer, {self.s: s, self.a_player: np.asarray(0).reshape(1,1)})
            a_plan_C = self.sess.run(self.action_layer, {self.s: s, self.a_player: np.asarray(1).reshape(1,1)})
            l = [a_plan_D,a_plan_C]
            if self.max_reward_strength is not None:
                l = [2 * self.max_reward_strength * (a_plan_X-0.5) for a_plan_X in l]
            # TODO redistribution only case (see 15 lines onwards)
            return np.transpose(np.reshape(np.asarray(l),[1,2]))
        else:
            # Planning actions in each of the 4 cases: DD, CD, DC, CC
            a_plan_DD = self.sess.run(self.action_layer, {self.s: s, self.a_player: np.array([0,0])[np.newaxis,:]})
            a_plan_CD = self.sess.run(self.action_layer, {self.s: s, self.a_player: np.array([1,0])[np.newaxis,:]})
            a_plan_DC = self.sess.run(self.action_layer, {self.s: s, self.a_player: np.array([0,1])[np.newaxis,:]})
            a_plan_CC = self.sess.run(self.action_layer, {self.s: s, self.a_player: np.array([1,1])[np.newaxis,:]})
            l_temp = [a_plan_DD,a_plan_CD,a_plan_DC,a_plan_CC]
            if self.max_reward_strength is not None:
                l0 = [2 * self.max_reward_strength * (a_plan_X[0,0]-0.5) for a_plan_X in l_temp]
                l1 = [2 * self.max_reward_strength * (a_plan_X[0,1]-0.5) for a_plan_X in l_temp]
            else:
                l0 = [a_plan_X[0,0] for a_plan_X in l_temp]
                l1 = [a_plan_X[0,1] for a_plan_X in l_temp]
            if self.revenue_neutral:
                l0 = [0.5*(elt[0]-elt[1]) for elt in zip(l0,l1)] 
        return np.transpose(np.reshape(np.asarray(l0),[2,2]))
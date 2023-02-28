import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def construct_q_function(agent_index, obs_n, act_n):
    '''
    Return information used to construct an agent's Q-function/critic network.
    :param agent_index: (int) Index of the agent.
    :param obs_n: (list) List of batched agent observations. Length: (num_agents).
        obs_n[i]: (tensorflow Tensor) Agent i's batched observations. Shape: (batch_size, observation_length).
    :param act_n: (list) List of batched agent actions. Length: (num_agents).
        act_n[i]: (tensorflow Tensor) Agent i's batched actions. Shape: (batch_size, action_dimension).

    :return q_input: (tensorflow Tensor) Batched input to the critic network for this agent. Shape: (batch_size, ???).
    :return num_q_outputs: (int) Number of outputs of the critic network.
    '''
    # TODO
    q_input = tf.concat([obs_n[agent_index], act_n[agent_index]], 1)
    num_q_outputs = 1
    return q_input, num_q_outputs


def compute_q_loss(q, target_q):
    '''
    Compute the loss used to train the Q-function/critic network over the batch of samples.
    :param q: (tensorflow Tensor) The batched Q-values for the agents' policies. Shape: (batch_size,). 
    :param target_q: (tensorflow Tensor) The batched target Q-values. Shape: (batch_size,). 

    :return q_loss: (tensorflow Tensor) Loss of the critic network. Shape: ().
    '''
    # TODO
    L2_q_y = tf.square(q - target_q)
    q_loss = tf.reduce_mean(L2_q_y)
    return q_loss


def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    assert not local_q_func, 'Not implemented'

    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input, num_q_outputs = construct_q_function(q_index, obs_ph_n, act_ph_n)
        q = q_func(q_input, num_q_outputs, scope="q_func", num_units=num_units)[:,0]
        q_loss = compute_q_loss(q, target_ph)

        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        loss = q_loss

        # apply gradient step to minimize loss wrt q_func_vars (params) using optimizer
        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping) 

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr]) # function that outputs q-loss, while updating q-params

        # target network
        target_q = q_func(q_input, num_q_outputs, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars) # update target-q with params of q-func

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q) # function takes as input q-inputs and outputs target-q-value

        return train, update_target_q, {'q_values': None, 'target_q_values': target_q_values}


def construct_p_function(agent_index, obs_n, action_dim):
    '''
    Return information used to construct an agent's policy/actor network.
    :param agent_index: (int) Index of the agent.
    :param obs_n: (list) List of batched agent observations. Length: (num_agents).
        obs_n[i]: (tensorflow Tensor) Agent i's batched observations. Shape: (batch_size, observation_length).
    :param action_dim: (int) Dimensionality of the action space.

    :return p_input: (tensorflow Tensor) Batched input to the actor network for this agent. Shape: (batch_size, ???).
    :return num_p_outputs: (int) Number of outputs of the actor network.
    '''
    # TODO
    p_input = obs_n[agent_index]
    num_p_outputs = action_dim
    return p_input, num_p_outputs


def compute_p_loss(q):
    '''
    Compute the loss used to train the policy/actor network over the batch of samples.
    :param q: (tensorflow Tensor) The batched Q-values for the agent's policy. Shape: (batch_size,). 

    :return pg_loss: (tensorflow Tensor) Loss of the actor network. Shape: ().
    '''
    # TODO
    pg_loss = tf.reduce_mean(q)
    return pg_loss


def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        action_dim = int(act_pdtype_n[p_index].param_shape()[0])
        p_input, num_p_outputs = construct_p_function(p_index, obs_ph_n, action_dim)

        p = p_func(p_input, num_p_outputs, scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample() # replace sample placeholder with actual sample from p-func
        q_input, num_q_outputs = construct_q_function(p_index, obs_ph_n, act_input_n)
        q = q_func(q_input, num_q_outputs, scope="q_func", reuse=True, num_units=num_units)[:,0]

        pg_loss = -compute_p_loss(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping) # update policy params

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr]) # outputs loss while updating params
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample) # get agent action
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, num_p_outputs, scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars) # update target params

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample) # get target action

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}


def compute_q_targets(agent_index, rew, done, obs_next_n, target_policy_func_n, target_q_func, gamma):
    '''
    Compute target Q-values used for training the agent's Q-function/critic network.
    :param agent_index: (int) Index of the agent.
    :param rew: (numpy.ndarray) Batched agent rewards. Shape: (batch_size,).
    :param done: (numpy.ndarray) Batched indicators of whether the episode is done. Shape: (batch_size,).
    :param obs_next_n: (list) List of batched agent observations for the next step. Length: (num_agents).
        obs_n[i]: (numpy.ndarray) Agent i's batched observations for the next step. Shape: (batch_size, observation_length).
    :param target_policy_func_n: (list) List of agent target policies.
        target_policy_func_n[i]: (function) Agent i's target policy. This function takes 1 argument (obs_i)
                regardless of how you implemented "construct_p_function" above.
    :param target_q_func: (function) This agent's target Q-function. This function takes 2N arguments in the order
            (obs_1, ..., obs_N, act_1, ..., act_N) regardless of how you implemented "construct_q_function" above.
    :param gamma: (float) Discount factor.

    :return target_q: (numpy.ndarray) Batched target Q-values for this agent. Shape: (batch_size,).
    '''
    # TODO
    # First, find each agent's target action for the next step.
    # target_q = 0.0
    # print("target_policy_func_n:",target_policy_func_n)
    # print("len(obs_next_n):",len(obs_next_n))
    
    action_next = []
    for i in range(len(obs_next_n)):
      action_next.append(target_policy_func_n[i](obs_next_n[i]))

    # Then, find the target Q-value for the next step. Note: If the episode is done, the target Q-value for the next step should be 0.
    target_q_next = target_q_func(*(obs_next_n + action_next))
    # Then, calculate the target Q-value for this step using the immediate reward and the discount factor.
    target_q = rew + gamma * target_q_next * (1.0 - done)
    return target_q


def update_target_param(param, target_param):
    '''
    Update a target network parameter.
    :param param: (tensorflow Variable) Parameter in actor or critic network. Shape: any.
    :param target_param: (tensorflow Variable) Corresponding parameter in the target networks. Shape: same as param.

    :return new_target_param: (tensorflow Variable) Updated value of target network parameter. Shape: same as param.
    '''
    # TODO
    new_target_param = 0.99 * target_param + 0.01 * param
    return new_target_param


def make_update_exp(vals, target_vals):
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        new_var_target = update_target_param(var, var_target)
        expression.append(var_target.assign(new_var_target))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # compute target q values
        target_policy_func_n = [agents[i].p_debug['target_act'] for i in range(self.n)]
        target_q_func = self.q_debug['target_q_values']
        target_q = compute_q_targets(self.agent_index, rew, done, obs_next_n, target_policy_func_n, target_q_func, self.args.gamma)

        # update q parameters
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # update p parameters
        p_loss = self.p_train(*(obs_n + act_n))

        # update target networks
        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), None, np.std(target_q)]

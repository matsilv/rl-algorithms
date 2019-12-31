#Author: Mattia Silvestri

import numpy as np
import tensorflow as tf
from common.policy import RandomPolicy, EpsilonGreedyPolicy, GreedyPolicy, StochasticPolicy
from common.memory import ReplayExperienceBuffer
from common.model import CNNModel, FullyConnectedModel, A2CNetwork
from common.atari_wrapper import make_env
import gym
import copy

# Tensorflow GPU setup
config = tf.ConfigProto()
# initial memory
# es.: 0.1 di 12 GB = 1.2 GB
config.gpu_options.per_process_gpu_memory_fraction = 0.2
# allow dynamic memory allocation
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"


# Deep Q-learning algorithm
def q_learning(batch, q_net, target_net, sess, gamma, input_shape, double):
    states = [val[0] for val in batch]
    next_states = [(np.zeros(q_net.num_states) if val[4] else val[2]) for val in batch]

    # predict Q(s,a) given the batch of states
    q_s_a = q_net.predict_batch(states, sess)
    # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
    q_s_a_d = target_net.predict_batch(next_states, sess)
    q_s_a_d_train = q_net.predict_batch(next_states, sess)

    # setup training arrays
    if type(input_shape) is tuple:
        x = np.zeros((len(batch), *q_net.num_states))
    else:
        x = np.zeros((len(batch), q_net.num_states))
    y = np.zeros((len(batch), q_net.num_actions))

    for i, b in enumerate(batch):
        state, action, reward, next_state = b[0], b[1], b[3], b[2]
        # get the current q values for all actions in state
        current_q = q_s_a[i]
        # update the q value for action
        if b[4]:
            # in this case, the game completed after action, so there is no max Q(s',a')
            # prediction possible
            current_q[action] = reward
        else:
            if not double:
                current_q[action] = reward + gamma * np.amax(q_s_a_d[i])
            else:
                best_next_action = np.argmax(q_s_a_d_train[i])
                current_q[action] = reward + gamma * q_s_a_d[i, best_next_action]
        x[i] = state
        y[i] = current_q

    loss = q_net.train_batch(sess, states=x, actions=None, q_vals=y)
    return loss


# Q-learning main function
def q_learning_main(args):
    ENV_NAME = args.env_name
    if args.render == 't':
        RENDER = True
    else:
        RENDER = False
    if args.atari == 't':
        ATARI = True
    else:
        ATARI = False
    NUM_STEPS = args.num_steps
    MEM_LENGHT = args.mem_size
    BATCH_SIZE = args.batch_size
    EPSILON_START = args.epsilon_start
    EPSILON_END = args.epsilon_end
    EPSILON_STEPS = args.epsilon_steps
    GAMMA = args.gamma
    TRAIN_INTERVAL = args.train_interval
    UPDATE_INTERVAL = args.update_interval
    if args.double == 't':
        DOUBLE = True
        print("Double Q-learning")
    else:
        DOUBLE = False
    if args.duel == 't':
        DUEL = True
        print("Dueling architecture")
    else:
        DUEL = False

    memory = ReplayExperienceBuffer(maxlen=MEM_LENGHT)
    # Reset the graph
    tf.reset_default_graph()
    if ATARI:
        env = make_env(ENV_NAME)
        INPUT_SHAPE = (84, 84, 4)
        q_fun_net = CNNModel(INPUT_SHAPE, env.action_space.n, BATCH_SIZE, scope="q_network", duel=DUEL)
        target_net = CNNModel(INPUT_SHAPE, env.action_space.n, BATCH_SIZE, scope="target_network", duel=DUEL)
    else:
        env = gym.make(ENV_NAME)
        INPUT_SHAPE = env.observation_space.shape[0]
        q_fun_net = FullyConnectedModel(INPUT_SHAPE, env.action_space.n, BATCH_SIZE, scope="q_network", duel=DUEL)
        target_net = FullyConnectedModel(INPUT_SHAPE, env.action_space.n, BATCH_SIZE, scope="target_network", duel=DUEL)

    print('Environment name: {}'.format(ENV_NAME))
    print('Observation space: {}'.format(env.observation_space))
    print('Action space: {}'.format(env.action_space))

    policy = EpsilonGreedyPolicy(env.action_space.n, EPSILON_START, EPSILON_END, EPSILON_STEPS)


    with tf.Session(config=config) as sess:
        sess.run(q_fun_net.var_init)
        sess.run(target_net.var_init)

        frames = 0
        r_interval = 0

        update_target_graph(sess)

        while frames < NUM_STEPS:
            game_over = False
            s_t = env.reset()
            score = 0
            loss = 0

            while not game_over:
                if RENDER:
                    env.render()

                q_values = q_fun_net.predict_one(s_t, sess)
                a_t = policy.select_action(q_values)
                s_tp1, r_t, game_over, _ = env.step(a_t)
                r_interval += r_t
                s_tp1 = np.array(s_tp1)
                memory.insert((s_t, a_t, s_tp1, r_t, game_over))
                s_t = s_tp1
                # print('Memory size: {}'.format(len(mem)))

                if len(memory) > BATCH_SIZE and frames % TRAIN_INTERVAL == 0:
                    loss = q_learning(memory.get_random_batch(BATCH_SIZE),
                                      q_fun_net, target_net, sess, GAMMA, INPUT_SHAPE,
                                      DOUBLE)

                score += r_t
                frames += 1

                if frames % UPDATE_INTERVAL == 0:
                    update_target_graph(sess)

                if frames % 10000 == 0:
                    r_interval /= 10000
                    print('Interval mean reward: {}'.format(r_interval))
                    r_interval = 0

            print('Frame: {}/{} | Score: {} | Epsilon: {} | Loss: {}'.
                  format(frames, NUM_STEPS, score, policy.epsilon, loss))

        s_t = env.reset()
        policy = GreedyPolicy(env.action_space.n)
        score = 0
        game_over = False

        while not game_over:
            q_values = q_fun_net.predict_one(s_t, sess)
            a_t = policy.select_action(q_values)
            s_tp1, r_t, game_over, _ = env.step(a_t)
            s_t = s_tp1
            score += r_t

        print('Score: {}'.format(score))

    env.close()
    sess.close()


# update target network
def update_target_graph(sess):
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "q_network")

    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "target_network")

    op_holder = []

    # Update our target_network parameters with DQNNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))

    sess.run(op_holder)

########################################################################################################################

# compute total expected reward
def calc_qvals(rewards, gamma):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


# Reinforce main function
def reinforce_main(args):
    ENV_NAME = args.env_name
    if args.render == 't':
        RENDER = True
    else:
        RENDER = False
    if args.atari == 't':
        ATARI = True
    else:
        ATARI = False
    NUM_STEPS = args.num_steps
    GAMMA = args.gamma

    # Reset the graph
    tf.reset_default_graph()
    if ATARI:
        env = make_env(ENV_NAME)
        INPUT_SHAPE = (84, 84, 4)
        policy_net = CNNModel(INPUT_SHAPE, env.action_space.n, scope="policy_network", apply_softmax=True)
    else:
        env = gym.make(ENV_NAME)
        INPUT_SHAPE = env.observation_space.shape[0]
        policy_net = FullyConnectedModel(INPUT_SHAPE, env.action_space.n, 1, scope="policy_network",
                                         apply_softmax=True)

    print('Environment name: {}'.format(ENV_NAME))
    print('Observation space: {}'.format(env.observation_space))
    print('Action space: {}'.format(env.action_space))

    policy = StochasticPolicy(env.action_space.n)

    with tf.Session(config=config) as sess:
        sess.run(policy_net.var_init)

        frames = 0
        r_interval = 0

        while frames < NUM_STEPS:
            game_over = False
            s_t = env.reset()
            score = 0
            loss = 0

            rewards = []
            actions = []
            states = []

            while not game_over:

                if RENDER:
                    env.render()

                probs = policy_net.predict_one(s_t, sess, apply_softmax=True)
                probs = probs.reshape(-1)
                a_t = policy.select_action(probs)
                action = np.zeros(env.action_space.n)
                action[a_t] = 1
                actions.append(action)
                states.append(s_t)
                s_tp1, r_t, game_over, _ = env.step(a_t)
                rewards.append(r_t)
                r_interval += r_t
                s_tp1 = np.array(s_tp1)
                s_t = s_tp1

                score += r_t
                frames += 1

                if frames % 10000 == 0:
                    r_interval /= 10000
                    print('Interval mean reward: {}'.format(r_interval))
                    r_interval = 0

            q_vals = calc_qvals(rewards, gamma=GAMMA)
            loss = policy_net.train_batch(sess, states, actions, q_vals)
            states.clear()
            q_vals.clear()

            print('Frame: {}/{} | Score: {} | Loss: {}'.
                  format(frames, NUM_STEPS, score, loss))

        s_t = env.reset()
        policy = StochasticPolicy(env.action_space.n)
        score = 0
        game_over = False

        while not game_over:
            probs = policy_net.predict_one(s_t, sess, apply_softmax=True)
            probs = probs.reshape(-1)
            a_t = policy.select_action(probs)
            s_tp1, r_t, game_over, _ = env.step(a_t)
            s_t = s_tp1
            score += r_t

        print('Score: {}'.format(score))

    env.close()
    sess.close()
########################################################################################################################

# A2C main function
def a2c_main(args):
    ENV_NAME = args.env_name
    if args.render == 't':
        RENDER = True
    else:
        RENDER = False
    if args.atari == 't':
        ATARI = True
    else:
        ATARI = False
    NUM_STEPS = args.num_steps
    GAMMA = args.gamma

    # Reset the graph
    tf.reset_default_graph()
    if ATARI:
        env = make_env(ENV_NAME)
        INPUT_SHAPE = (84, 84, 4)
        policy_net = A2CNetwork(INPUT_SHAPE, env.action_space.n, 1, scope="policy_network", cnn=True)
    else:
        env = gym.make(ENV_NAME)
        INPUT_SHAPE = env.observation_space.shape[0]
        policy_net = A2CNetwork(INPUT_SHAPE, env.action_space.n, 1, scope="policy_network", cnn=False)

    BATCH_SIZE = args.batch_size

    print('Environment name: {}'.format(ENV_NAME))
    print('Observation space: {}'.format(env.observation_space))
    print('Action space: {}'.format(env.action_space))

    policy = StochasticPolicy(env.action_space.n)

    with tf.Session(config=config) as sess:
        sess.run(policy_net.var_init)

        frames = 0
        r_interval = 0

        rewards = []
        actions = []
        states = []
        idx = 0

        loss_policy = 0
        loss_value = 0

        while frames < NUM_STEPS:
            game_over = False
            s_t = env.reset()
            score = 0

            while not game_over:

                if RENDER:
                    env.render()

                probs = policy_net.predict_one(s_t, sess, apply_softmax=True, cnn=ATARI)
                probs = probs.reshape(-1)
                a_t = policy.select_action(probs)
                action = np.zeros(env.action_space.n)
                action[a_t] = 1
                actions.append(action)
                states.append(s_t)
                s_tp1, r_t, game_over, _ = env.step(a_t)
                rewards.append(r_t)
                r_interval += r_t
                s_tp1 = np.array(s_tp1)
                s_t = s_tp1

                score += r_t
                frames += 1

                if frames % 10000 == 0:
                    r_interval /= 10000
                    print('Interval mean reward: {}'.format(r_interval))
                    r_interval = 0

            q_vals = calc_qvals(rewards, gamma=GAMMA)
            loss_policy, loss_value = policy_net.train_batch(sess, states, actions, q_vals)
            states.clear()
            q_vals.clear()
            rewards.clear()
            actions.clear()

            print('Frame: {}/{} | Score: {} | Loss policy: {} | Loss value: {}'.
                  format(frames, NUM_STEPS, score, loss_policy, loss_value))

        s_t = env.reset()
        policy = StochasticPolicy(env.action_space.n)
        score = 0
        game_over = False

        while not game_over:
            probs = policy_net.predict_one(s_t, sess)
            a_t = policy.select_action(probs.reshape(-1))
            s_tp1, r_t, game_over, _ = env.step(a_t)
            s_t = s_tp1
            score += r_t

        print('Score: {}'.format(score))

    env.close()
    sess.close()


#Author: Mattia Silvestri

import argparse
from common.algorithms import q_learning_main, reinforce_main, a2c_main


""" main program """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, help="environment name")
    parser.add_argument("--render", type=str, default='f', help="t if you want to render the environment, f otherwise")
    parser.add_argument("--atari", type=str, default='f', help="t if you want to train on atari env, f otherwise")
    parser.add_argument("--num_steps", type=int, default=1000000, help="number of episodes")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="epsilon initial value")
    parser.add_argument("--epsilon_end", type=float, default=0.1, help="epsilon final value")
    parser.add_argument("--epsilon_steps", type=int, default=900000, help="annealing steps for epsilon")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--train_interval", type=int, default=4, help="train interval")
    parser.add_argument("--update_interval", type=int, default=10000, help="target network update interval")
    parser.add_argument("--double", type=str, default='f', help="t if you want to enable Double Q-learning")
    parser.add_argument("--duel", type=str, default='f', help="t if you want to enable Dueling architecture")
    parser.add_argument("--mem_size", type=int, default=50000, help="replay buffer size")

    args = parser.parse_args()

    #q_learning_main(args)
    #reinforce_main(args)
    a2c_main(args)

    exit(0)





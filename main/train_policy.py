import os
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger

import gym

def train(env_id, num_timesteps, seed, model_path=None):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = make_mujoco_env(env_id, seed)

    # parameters below were the best found in a simple random search
    # these are good enough to make humanoid walk, but whether those are
    # an absolute best or not is not certain
    pi = pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, 
            optim_stepsize=3e-4, 
            optim_batchsize=64, 
            gamma=0.99, 
            lam=0.95,
            schedule='linear',
        )
    print(pi)
    env.close()
    if model_path:
        U.save_state(model_path)
        
    return pi

def main():
    logger.configure()
    parser = mujoco_arg_parser()
    parser.add_argument('--model-path', default=os.path.join(logger.get_dir(), 'policy'))
    parser.set_defaults(num_timesteps=int(2e7))
   
    args = parser.parse_args()
    
    if not args.play:
        # train the model
        train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, model_path=args.model_path)
    else:       
        # construct the model object, load pre-trained model and render
        pi = train(args.env, num_timesteps=1, seed=args.seed)
        U.load_state(args.model_path)
        env = make_mujoco_env(args.env, seed=0)

        ob = env.reset()        
        while True:
            action = pi.act(stochastic=False, ob=ob)[0]
            ob, _, done, _ =  env.step(action)
            print(ob,action)
            #env.render()
            if done:
                ob = env.reset()
        
if __name__ == '__main__':
    main()

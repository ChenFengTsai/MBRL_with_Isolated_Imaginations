import argparse
import collections
import functools
import os
import pathlib
import sys
import warnings
import numpy as np
import ruamel.yaml as yaml
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(pathlib.Path(__file__).parent))

import torch
import torch.nn.functional as F
from torch import distributions as torchd
from torch import nn

# Import SVEA and related modules
from algorithms.sac import SAC
from algorithms.svea import SVEA, random_conv
from replay_buffer import ReplayBuffer
import utils
import tools
import wrappers
from config_saver import save_config
import gym

to_np = lambda x: x.detach().cpu().numpy()

original_box_init = gym.spaces.Box.__init__

def patched_box_init(self, low, high, shape=None, dtype=np.float32, seed=None):
    if shape is None and np.isscalar(low) and np.isscalar(high):
        shape = ()  # Scalar shape for single values
    return original_box_init(self, low, high, shape, dtype, seed)

gym.spaces.Box.__init__ = patched_box_init

class SVEAAgent:
    """Wrapper for SVEA algorithm to work with the environment framework"""
    
    def __init__(self, config, logger, obs_shape, action_shape, action_range):
        self._config = config
        self._logger = logger
        
        if action_range is not None:
            if isinstance(action_range[0], np.ndarray):
                action_range = [
                    torch.tensor(action_range[0], dtype=torch.float32, device=config.device),
                    torch.tensor(action_range[1], dtype=torch.float32, device=config.device)
                ]
        # Initialize SVEA directly
        self._svea = SVEA(obs_shape, action_shape, action_range, config)
        
        # Create replay buffer
        self._replay_buffer = ReplayBuffer(
            obs_shape=obs_shape,
            action_shape=action_shape,
            capacity=config.replay_buffer_capacity,
            image_pad=getattr(config, 'image_pad', 4),
            device=config.device,
            ted=config.ted
        )
        
        self._step = 0
        
    def act(self, obs, sample=True):
        """Simple action selection"""
        return self._svea.act(obs, sample=sample)
        
    def update(self, replay_buffer, logger, step):
        """Update the agent"""
        return self._svea.update(replay_buffer, logger, step)
        
    def train(self, training=True):
        """Set training mode"""
        self._svea.train(training)
        
    def state_dict(self):
        """Get state dict for saving"""
        return {
            'actor': self._svea.actor.state_dict(),
            'critic': self._svea.critic.state_dict(),
            'critic_target': self._svea.critic_target.state_dict(),
            'actor_optimizer': self._svea.actor_optimizer.state_dict(),
            'critic_optimizer': self._svea.critic_optimizer.state_dict(),
            'log_alpha': self._svea.log_alpha,
            'log_alpha_optimizer': self._svea.log_alpha_optimizer.state_dict(),
            'step': self._step,
            'episode_id': self._episode_id
        }
        
    def load_state_dict(self, state_dict):
        """Load state dict"""
        self._svea.actor.load_state_dict(state_dict['actor'])
        self._svea.critic.load_state_dict(state_dict['critic'])
        self._svea.critic_target.load_state_dict(state_dict['critic_target'])
        self._svea.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self._svea.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        self._svea.log_alpha = state_dict['log_alpha']
        self._svea.log_alpha_optimizer.load_state_dict(state_dict['log_alpha_optimizer'])
        self._step = state_dict['step']
        self._episode_id = state_dict['episode_id']
        
        if self._svea.ted:
            self._svea.ted_classifier.load_state_dict(state_dict.get('ted_classifier', {}))
            self._svea.ted_optimizer.load_state_dict(state_dict.get('ted_optimizer', {}))


def count_steps(folder):
    return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))


def make_env(config, logger, mode):
    """Create environment - adapted from original code"""
    suite, task = config.task.split('_', 1)
    
    if suite == 'dmc':
        env = wrappers.DeepMindControl(task, config.action_repeat, config.size)
        env = wrappers.NormalizeActions(env)
    elif suite == 'dmcbg':
        env = wrappers.DeepMindControlGen(task, config.seed, config.action_repeat, config.size, config.eval_mode)
        env = wrappers.NormalizeActions(env)    
    elif suite == 'atari':
        env = wrappers.Atari(
            task, config.action_repeat, config.size,
            grayscale=config.grayscale,
            life_done=False and ('train' in mode),
            sticky_actions=True,
            all_actions=True)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key='action')
    env = wrappers.RewardObs(env)
    
    return env


def process_episode(config, logger, mode, episode):
    """Process completed episodes"""
    length = len(episode['reward']) - 1
    score = float(episode['reward'].astype(np.float64).sum())
    
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    
    if mode == 'eval':
        video = episode['image']
        logger.video(f'{mode}_policy', video[None])
    
    logger.write()


def parse_svea_args():
    """Parse arguments using the same structure as the original Dreamer config system"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()
    
    # Load configs from YAML file
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
    defaults = {}
    for name in args.configs:
        defaults.update(configs[name])
    
    # Create parser with all config options
    parser = argparse.ArgumentParser()

    # Add all config parameters - maintaining original structure
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    
    args = parser.parse_args(remaining)
    
    # Process device
    if isinstance(args.device, str):
        args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Handle size parameter (can be int or list)
    if hasattr(args, 'size'):
        if isinstance(args.size, int):
            args.size = [args.size, args.size]
    else:
        args.size = [64, 64]  # Default size
    
    # Map original parameter names to SVEA equivalents
    if not hasattr(args, 'replay_buffer_capacity'):
        args.replay_buffer_capacity = 100000
    if not hasattr(args, 'num_seed_steps'):
        args.num_seed_steps = getattr(args, 'prefill', 1000)
    if not hasattr(args, 'num_train_iters'):
        args.num_train_iters = getattr(args, 'train_steps', 1)
    if not hasattr(args, 'num_eval_episodes'):
        args.num_eval_episodes = 10
    if not hasattr(args, 'save_freq'):
        args.save_freq = 50000
    if not hasattr(args, 'image_pad'):
        args.image_pad = 4
    
    # Map TED parameters
    if not hasattr(args, 'ted'):
        args.ted = getattr(args, 'ted_mode', False)
    if not hasattr(args, 'ted_coef'):
        args.ted_coef = getattr(args, 'ted_coefficient', 1.0)
    if not hasattr(args, 'ted_lr'):
        args.ted_lr = getattr(args, 'model_lr', 1e-3)
    
    # Map learning rates if not specified
    if not hasattr(args, 'critic_lr'):
        args.critic_lr = getattr(args, 'value_lr', 1e-3)
    if not hasattr(args, 'actor_lr'):
        args.actor_lr = getattr(args, 'actor_lr', 1e-3)
    if not hasattr(args, 'alpha_lr'):
        args.alpha_lr = 1e-4
    
    # Map network architecture parameters - keep original structure as provided
    if not hasattr(args, 'hidden_dim'):
        args.hidden_dim = 1024
    if not hasattr(args, 'hidden_depth'):
        args.hidden_depth = 2
    if not hasattr(args, 'num_filters'):
        args.num_filters = 32
    if not hasattr(args, 'feature_dim'):
        args.feature_dim = 50
    if not hasattr(args, 'num_conv_layers'):
        args.num_conv_layers = 4
    
    # Map other SAC parameters - keep original structure as provided
    if not hasattr(args, 'actor_update_freq'):
        args.actor_update_freq = 2
    if not hasattr(args, 'critic_target_update_freq'):
        args.critic_target_update_freq = 2
    if not hasattr(args, 'critic_tau'):
        args.critic_tau = 0.01
    if not hasattr(args, 'encoder_tau'):
        args.encoder_tau = 0.01
    if not hasattr(args, 'init_temperature'):
        args.init_temperature = 0.1
    if not hasattr(args, 'actor_log_std_min'):
        args.actor_log_std_min = -10.0
    if not hasattr(args, 'actor_log_std_max'):
        args.actor_log_std_max = 2.0
    
    return args


def main(config):
    # Setup directories
    logdir = pathlib.Path(config.logdir).expanduser()
    if logdir.exists():
        # Create unique directory with suffix
        parent_dir = logdir.parent
        base_name = logdir.name
        existing_dirs = [d for d in parent_dir.glob(f"{base_name}_*") if d.is_dir()]
        max_suffix = 0
        for dir_path in existing_dirs:
            suffix_str = dir_path.name[len(base_name)+1:]
            if suffix_str.isdigit():
                max_suffix = max(max_suffix, int(suffix_str))
        new_suffix = max_suffix + 1
        logdir = pathlib.Path(f"{logdir}_{new_suffix}")
        print(f"Logdir already exists. Using {logdir} instead.")
    
    config.logdir = str(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    
    # Adjust config for action repeat
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    
    print('Logdir', logdir)
    logger = tools.Logger(logdir, config.action_repeat * 0)
    
    # Create environment
    print('Create env.')
    train_env = make_env(config, logger, 'train')
    eval_env = make_env(config, logger, 'eval')
                 
    # Get action and observation specs
    acts = train_env.action_space
    obs_space = train_env.observation_space
    
    # Setup action dimensions
    if hasattr(acts, 'n'):
        action_dim = acts.n
        action_range = None
    else:
        action_dim = acts.shape[0]
        action_range = [acts.low, acts.high]
    
    config.action_dim = action_dim
    config.num_actions = action_dim
    config.action_range = action_range
    
    # Setup observation shape
    if isinstance(obs_space, gym.spaces.Dict):
        if 'image' in obs_space.spaces:
            obs_shape = obs_space['image'].shape
            print("Image observation shape:", obs_shape)  # Should print (64, 64, 3)
        else:
            raise ValueError("No 'image' key found in observation space")
    else:
        obs_shape = obs_space.shape
        print("Direct observation shape:", obs_shape)

    
    # Create SVEA agent
    print('Create SVEA agent.')
    agent = SVEAAgent(config, logger, obs_shape, (action_dim,), action_range)
    
    # Training loop (simplified, matching original SVEA)
    episode = 0
    step = 0
    
    while step < config.steps:
        # Reset environment
        obs = train_env.reset()
        # Extract image if obs is a dictionary
        if isinstance(obs, dict):
            obs = obs['image']
            
        done = False
        episode_reward = 0
        episode_step = 0
        
        while not done and step < config.steps:
            # Select action
            if step < config.num_seed_steps:
                action = train_env.action_space.sample()
            else:
                with torch.no_grad():
                    # obs should now be just the image array
                    action = agent.act(obs, sample=True)
            
            # Take step
            if isinstance(action, (np.ndarray, torch.Tensor)):
                action_dict = {'action': action}
            else:
                action_dict = action
            next_obs, reward, done, info = train_env.step(action_dict)
            
            # Extract image from next_obs if it's a dictionary
            if isinstance(next_obs, dict):
                next_obs_image = next_obs['image']
            else:
                next_obs_image = next_obs
                
            # Convert from HWC to CHW format
            if len(next_obs_image.shape) == 3 and next_obs_image.shape[-1] == 3:
                next_obs_image = next_obs_image.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
            # Store in replay buffer
            agent._replay_buffer.add(obs, action, reward, next_obs_image, done, done, episode)
            
            # Update agent
            if step >= config.num_seed_steps:
                for _ in range(config.num_train_iters):
                    agent.update(agent._replay_buffer, logger, step)
            
            # Update obs for next iteration - this is crucial!
            obs = next_obs_image  # Use the extracted image, not the full dict
            episode_reward += reward
            episode_step += 1
            step += 1
            
            # Evaluation
            if step % config.eval_every == 0:
                print(f'Evaluating at step {step}')
                eval_rewards = []
                for _ in range(config.num_eval_episodes):
                    eval_obs = eval_env.reset()
                    # Extract image for evaluation too
                    if isinstance(eval_obs, dict):
                        eval_obs = eval_obs['image']
                        
                    eval_done = False
                    eval_reward = 0
                    
                    while not eval_done:
                        with torch.no_grad():
                            eval_action = agent.act(eval_obs, sample=False)
                        
                        if isinstance(eval_action, (np.ndarray, torch.Tensor)):
                            eval_action_dict = {'action': eval_action}
                        else:
                            eval_action_dict = eval_action
                            
                        next_eval_obs, r, eval_done, _ = eval_env.step(eval_action_dict)
                        
                        # Extract image for next iteration
                        if isinstance(next_eval_obs, dict):
                            eval_obs = next_eval_obs['image']
                        else:
                            eval_obs = next_eval_obs
                            
                        eval_reward += r
                    
                    eval_rewards.append(eval_reward)
                
                avg_reward = np.mean(eval_rewards)
                print(f'Step {step}: Average reward = {avg_reward}')
                logger.scalar('eval_reward', avg_reward)
            
            # Save model
            if step % config.save_freq == 0:
                torch.save(agent.state_dict(), logdir / f'svea_model_{step}.pt')
        
        episode += 1
        print(f'Episode {episode}: Reward = {episode_reward}, Steps = {episode_step}')
    
    # Final save
    torch.save(agent.state_dict(), logdir / 'final_model.pt')
    
    # Cleanup
    train_env.close()
    eval_env.close()


if __name__ == '__main__':
    # Parse arguments using the original structure
    config = parse_svea_args()
    
    # Set random seed
    tools.set_seed_everywhere(config.seed)
    
    main(config)
import wandb
import argparse
import os
import torch
import numpy as np
import gym  # type: ignore
import pandas as pd  # type: ignore
import sys

sys.path.append("..")
from dreamerv2.utils.wrapper import (
    GymMinAtar,
    OneHotAction,
    breakoutPOMDP,
    space_invadersPOMDP,
    seaquestPOMDP,
    asterixPOMDP,
    freewayPOMDP,
)
from dreamerv2.utils.caption_generation import PomdpBreakoutCaptioner
from dreamerv2.training.config import MinAtarConfig
from dreamerv2.training.trainer import Trainer
from dreamerv2.training.evaluator import Evaluator

pomdp_wrappers = {
    "breakout": breakoutPOMDP,
    "seaquest": seaquestPOMDP,
    "space_invaders": space_invadersPOMDP,
    "asterix": asterixPOMDP,
    "freeway": freewayPOMDP,
}


def main(args):
    wandb.login()
    env_name = args.env
    exp_id = args.id + "_pomdp"

    """make dir for saving results"""
    result_dir = os.path.join("results", "{}_{}".format(env_name, exp_id))
    model_dir = os.path.join(result_dir, "models")  # dir to save learnt models
    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device:
        device = torch.device("cuda")
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device("cpu")
    print("using :", device)

    PomdpWrapper = pomdp_wrappers[env_name]
    env = PomdpWrapper(OneHotAction(GymMinAtar(env_name)))
    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]
    obs_dtype = bool
    action_dtype = np.float32
    batch_size = args.batch_size
    seq_len = args.seq_len

    config = MinAtarConfig(
        env=env_name,
        obs_shape=obs_shape,
        action_size=action_size,
        obs_dtype=obs_dtype,
        action_dtype=action_dtype,
        seq_len=seq_len,
        batch_size=batch_size,
        model_dir=model_dir,
    )
    captioner = PomdpBreakoutCaptioner(env)
    config_dict = config.__dict__
    trainer = Trainer(config, device)
    evaluator = Evaluator(config, device)
    training_step = 0
    image_dir = "/home/mattbarker/dev/dreamerv2/training_images/"
    image_paths = []
    image_captions = []
    with wandb.init(project="mastering MinAtar with world models", config=config_dict):
        """training loop"""
        print("...training...")
        train_metrics = {}
        trainer.collect_seed_episodes(env)
        # Obs is array of (paddle, ball, bricks)
        obs, score = env.reset(), 0
        done = False
        prev_rssmstate = trainer.RSSM._init_rssm_state(1)
        prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
        episode_actor_ent = []
        scores = []
        best_mean_score = 0
        train_episodes = 0
        best_save_path = os.path.join(model_dir, "models_best.pth")
        for iter in range(1, trainer.config.train_steps):
            if iter % trainer.config.train_every == 0:
                train_metrics = trainer.train_batch(train_metrics)
            if iter % trainer.config.slow_target_update == 0:
                trainer.update_target()
            if iter % trainer.config.save_every == 0:
                trainer.save_model(iter)
            with torch.no_grad():
                embed = trainer.ObsEncoder(
                    torch.tensor(obs, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(trainer.device)
                )
                _, posterior_rssm_state = trainer.RSSM.rssm_observe(
                    embed, prev_action, not done, prev_rssmstate
                )
                model_state = trainer.RSSM.get_model_state(posterior_rssm_state)
                action, action_dist = trainer.ActionModel(model_state)
                action = trainer.ActionModel.add_exploration(action, iter).detach()
                action_ent = torch.mean(action_dist.entropy()).item()
                episode_actor_ent.append(action_ent)

            next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
            score += rew

            if done:
                image_captions.append(captioner.generate_caption(next_obs))
                env.env.env.env.display_state(
                    0, fp=image_dir + str(training_step) + ".png"
                )
                image_paths.append(image_dir + str(training_step) + ".png")
                train_episodes += 1
                trainer.buffer.add(obs, action.squeeze(0).cpu().numpy(), rew, done)
                train_metrics["train_rewards"] = score
                train_metrics["action_ent"] = np.mean(episode_actor_ent)
                train_metrics["train_steps"] = iter
                wandb.log(train_metrics, step=train_episodes)
                scores.append(score)
                if len(scores) > 100:
                    scores.pop(0)
                    current_average = np.mean(scores)
                    if current_average > best_mean_score:
                        best_mean_score = current_average
                        print("saving best model with mean score : ", best_mean_score)
                        save_dict = trainer.get_save_dict()
                        torch.save(save_dict, best_save_path)

                obs, score = env.reset(), 0
                done = False
                prev_rssmstate = trainer.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
                episode_actor_ent = []
            else:
                trainer.buffer.add(
                    obs, action.squeeze(0).detach().cpu().numpy(), rew, done
                )
                obs = next_obs
                image_captions.append(captioner.generate_caption(obs))
                env.env.env.env.display_state(
                    0, fp=image_dir + str(training_step) + ".png"
                )
                image_paths.append(image_dir + str(training_step) + ".png")
                prev_rssmstate = posterior_rssm_state
                prev_action = action
            if training_step == 0 or training_step % 1000 == 0:
                df = pd.DataFrame(
                    {"image_path": image_paths, "caption": image_captions}
                )
                df.to_csv(
                    os.path.join(
                        "/home/mattbarker/dev/dreamerv2/",
                        "breakout_captions.csv",
                    ),
                    index=False,
                )
            training_step += 1

    """evaluating probably best model"""
    evaluator.eval_saved_agent(env, best_save_path)


if __name__ == "__main__":
    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help="mini atari env name")
    parser.add_argument("--id", type=str, default="0", help="Experiment ID")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--device", default="cuda", help="CUDA or CPU")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument(
        "--seq_len", type=int, default=50, help="Sequence Length (chunk length)"
    )
    args = parser.parse_args()
    main(args)

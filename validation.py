from typing import Optional, Type
import habitat_sim.utils
from habitat import Config, Dataset
import cv2
from gym.spaces.dict import Dict as SpaceDict
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from habitat.core.spaces import ActionSpace, EmptySpace
import numpy as np
from env_utils.custom_habitat_env import RLEnv, MIN_DIST, MAX_DIST
import habitat
from habitat.utils.visualizations.utils import images_to_video
from env_utils.custom_habitat_map import TopDownGraphMap
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import env_utils.noisy_actions
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.utils.common import quat_to_coeffs
import imageio
import os
import time
import pickle
import quaternion as q
import scipy
import torch
from typing import Dict, List, Optional, Tuple
from OVRL_model import OVRL
from OVRL_LSTM_model import OVRL_LSTM
from vit_model import ViTForNavigation

def observations_to_image(observation: Dict, mode='panoramic', local_imgs=None, clip=None, center_agent=True) -> np.ndarray:
    size = 2.0
    egocentric_view = []

    rgb = observation['panoramic_rgb']
    if not isinstance(rgb, np.ndarray):
        rgb = rgb.cpu().numpy()
    rgb = cv2.putText(np.ascontiguousarray(rgb), 'current_obs', (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
    egocentric_view.append(rgb)

    goal_rgb = (observation['target_goal'] * 255)
    if not isinstance(goal_rgb, np.ndarray):
        goal_rgb = goal_rgb.cpu().numpy()

    if len(goal_rgb.shape) == 4:
        goal_rgb = np.concatenate(np.split(goal_rgb[:, :, :, :3], goal_rgb.shape[0], axis=0), 1).squeeze(axis=0)
    else:
        goal_rgb = goal_rgb[:, :, :3]
    goal_rgb = cv2.putText(np.ascontiguousarray(goal_rgb), 'target_obs', (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
    egocentric_view.append(goal_rgb.astype(np.uint8))

    if len(egocentric_view) > 0:
        if mode == 'panoramic':
            egocentric_view = np.concatenate(egocentric_view, axis=0)
        else:
            egocentric_view = np.concatenate(egocentric_view, axis=1)

        frame = cv2.resize(egocentric_view, dsize=None, fx=size * 0.75, fy=size)
    else:
        frame = None

    return frame

optimal_path_length = None

if __name__ == '__main__':
    from env_utils.make_env_utils import add_panoramic_camera
    from configs.default import get_config
    import numpy as np
    import os
    import time
    from habitat import make_dataset
    from env_utils.task_search_env import SearchEnv
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    config = get_config()
    config.defrost()
    config.DIFFICULTY = 'medium'
    habitat_api_path = os.path.dirname(habitat.__file__)
    config.TASK_CONFIG.DATASET.SCENES_DIR = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.SCENES_DIR)
    config.TASK_CONFIG.DATASET.DATA_PATH = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.DATA_PATH)
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 20

    ### ADD
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

    config.NUM_PROCESSES = 1
    config.NUM_VAL_PROCESSES = 0
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    training_scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        training_scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
    # config.TASK_CONFIG.DATASET.CONTENT_SCENES = ['Denmark','Ribera','Sands','Cantwell','Eudora','Sisters']
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = ['Adrian']
    config.TASK_CONFIG = add_panoramic_camera(config.TASK_CONFIG)
    config.record = True
    config.freeze()
    action_list = config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS

    env = SearchEnv(config)
    env.habitat_env._sim.seed(1121)
    obs = env.reset()
    env.build_path_follower()
    done = False
    fps = {}
    reset_time = {}

    scene = env.habitat_env.current_episode.scene_id.split('/')[-2]
    fps[scene] = []
    reset_time[scene] = []
    action = 0
    save_node = 0
    model = ViTForNavigation()
    sd = torch.load('epoch0005iter15500.pt')
    model.load_state_dict(sd['state_dict'])

    total_episodes = 0
    successful_episodes = 0
    total_spl = 0.0
    total_distance_to_goal = 0.0

    dts_total = 0.0
    success_threshold = 1.0
    min_distance_to_goal = float('inf')
    actual_path_length = 0.0
    last_position = env.habitat_env._sim.get_agent_state().position

    follower = ShortestPathFollower(env.habitat_env._sim, success_threshold, False)

    while True:
        if optimal_path_length is None:
            goal_pos = env.habitat_env.current_episode.goals[0].position
            optimal_path_length = env.habitat_env._sim.geodesic_distance(
                env.habitat_env._sim.get_agent_state().position,
                goal_pos
            )

        best_action = env.get_best_action()
        '''img = observations_to_image(obs)
        img = cv2.resize(img, dsize=(400, 280))
        cv2.imshow('render', img[:, :, [2, 1, 0]])
        cv2.waitKey(5)'''

        tic = time.time()

        metrics = env.get_metrics()
        distance_to_goal = metrics['distance_to_goal']
        min_distance_to_goal = min(min_distance_to_goal, distance_to_goal)

        current_position = env.habitat_env._sim.get_agent_state().position
        actual_path_length += np.linalg.norm(current_position - last_position)
        last_position = current_position
        print(f"Action: {action}, Actual Path Length: {actual_path_length}, Optimal Path Length: {optimal_path_length}, Distance To Goal: {distance_to_goal}")


        if distance_to_goal < success_threshold:
            obs, reward, done, info = env.step(0)  # 执行停止动作
            metrics = env.get_metrics()
            spl = metrics['spl']

        else:
            if action == 0:
                action = 1
            obs, reward, done, info = env.step(action)
            state = torch.tensor(obs['panoramic_rgb'], dtype=torch.float32).unsqueeze(0)
            target = torch.tensor(obs['target_goal'], dtype=torch.float32).unsqueeze(0)
            action, feature = model(state, target)
            action = action.item()
            spl = 0.0

        if done:
            total_episodes += 1
            if metrics['success'] == 1.0:
                successful_episodes += 1
            total_spl += spl
            total_distance_to_goal += distance_to_goal

            dts = max(min_distance_to_goal - success_threshold, 0)
            dts_total += dts

            current_sr = successful_episodes / total_episodes
            current_avg_spl = total_spl / total_episodes
            current_avg_dts = dts_total / total_episodes

            print(f"Episode {total_episodes} ended.")
            print(f"Success: {metrics['success']}")
            print(f"Current SR: {current_sr}")
            print(f"Current SPL: {spl}")
            print(f"Average SPL: {current_avg_spl}")
            print(f"Current DTS: {dts}")
            print(f"Average DTS: {current_avg_dts}")

            tic = time.time()
            obs = env.reset()

            optimal_path_length = None

            toc = time.time()
            scene = env.habitat_env.current_episode.scene_id.split('/')[-2]
            fps[scene] = []
            reset_time[scene] = []
            reset_time[scene].append(toc - tic)
            done = False
            min_distance_to_goal = float('inf')  # reset
            actual_path_length = 0.0
            last_position = env.habitat_env._sim.get_agent_state().position  # reset
        else:
            toc = time.time()
            fps[scene].append(toc - tic)




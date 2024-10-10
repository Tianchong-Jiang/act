import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import pickle

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats

        # get object name
        filename = os.listdir(dataset_dir)[0]
        self.object_name = filename.split('_')[0]


    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'{self.object_name}_{episode_id}.pt')
        with open(dataset_path, 'rb') as f:
            # filter out episodes with less than 6 joint states
            root = pickle.load(f)
            joint_states = [elem['joint_states'] for elem in root]
            keep_indices = [i for i in range(0, len(joint_states)) if len(joint_states[i]) == 6]
            root = [root[i] for i in keep_indices]
            # episode_len = len(root)
            episode_len = 70 # TODO: TC: temp
            start_ts = np.random.choice(episode_len)

            joint_states = np.array([elem['joint_states'] for elem in root])
            grasp = np.array([elem['cmd_grasp_pos'] for elem in root])
            joint_and_grasp = np.concatenate([joint_states, grasp[..., None] ], axis=1)
            joint_and_grasp = joint_and_grasp[:70] # TODO: TC: temp
            qpos = joint_and_grasp[0]

            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[start_ts][cam_name]

            # get all actions after and including start_ts
            action = joint_and_grasp[start_ts+1:]
            action_len = episode_len - start_ts - 1

        padded_action = np.zeros_like(joint_and_grasp, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos)
        action_data = torch.from_numpy(padded_action)
        is_pad = torch.from_numpy(is_pad)

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data.float(), qpos_data.float(), action_data.float(), is_pad.bool()


def get_norm_stats(dataset_dir, num_episodes):
    filename = os.listdir(dataset_dir)[0]
    object_name = filename.split('_')[0]

    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'{object_name}_{episode_idx}.pt')
        with open(dataset_path, 'rb') as f:
            root = pickle.load(f)
            joint_states = [elem['joint_states'] for elem in root]
            keep_indices = [i for i in range(0, len(joint_states)) if len(joint_states[i]) == 6]
            root = [root[i] for i in keep_indices]

            joint_states = np.array([elem['joint_states'] for elem in root])
            grasp = np.array([elem['cmd_grasp_pos'] for elem in root])
            qpos = np.concatenate([joint_states, grasp[..., None] ], axis=1)
            qpos = qpos[:70] # TODO: TC: temp
            action = qpos
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    dataset_dir = '/mount/sept30_white'

    # dataset = EpisodicDataset([0], dataset_dir, ['camarm', 'camouter'], 1)
    # data = dataset[0]

    train_dataloader, val_dataloader, norm_stats = load_data(dataset_dir, 50, ['camarm', 'camouter'], 1, 1)
    for i, data in enumerate(train_dataloader):
        image_data, qpos_data, action_data, is_pad = data
        print(image_data.shape, qpos_data.shape, action_data.shape, is_pad.shape)
        break

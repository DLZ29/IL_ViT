import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import imageio
import cv2
import copy


class BC_trainer(nn.Module):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.torch_device = 'cuda:0'
        self.optim = optim.Adam(
            list(filter(lambda p: p.requires_grad,self.agent.parameters())),
            lr=0.0025
        )


    def save(self,file_name=None, epoch=0, step=0):
        if file_name is not None:
            save_dict = {}

            save_dict['trained'] = [epoch, step]
            save_dict['state_dict'] = self.agent.state_dict()
            torch.save(save_dict, file_name)

    def forward(self, batch, train=True):
        demo_rgb, demo_depth, demo_act, positions, rotations, targets, target_img, scene, start_pose, aux_info = batch
        demo_rgb, demo_depth, demo_act = demo_rgb.to(self.torch_device), demo_depth.to(self.torch_device), demo_act.to(self.torch_device)
        target_img, positions, rotations = target_img.to(self.torch_device), positions.to(self.torch_device), rotations.to(self.torch_device)
        #aux_info = {'have_been': aux_info['have_been'].to(self.torch_device),
         #           'distance': aux_info['distance'].to(self.torch_device)}
        self.B = demo_act.shape[0]


        lengths = (demo_act > -10).sum(dim=1)

        T = lengths.max().item()
        #hidden_states = torch.zeros(self.agent.net.num_recurrent_layers, self.B, self.agent.net._hidden_size).to(self.torch_device)
        actions = torch.zeros([self.B]).cuda()
        results = {'imgs': [], 'curr_node': [], 'node_list':[], 'actions': [], 'gt_actions': [], 'target': [], 'scene':scene[0], 'A': [], 'position': [],
                   'have_been': [], 'distance': [], 'pred_have_been': [], 'pred_distance': []}
        losses = []


        for t in range(T):
            current_timestep = t
            total_timestep = T
            masks = lengths > t
            if t == 0: masks[:] = False
            target_goal = target_img[torch.range(0,self.B-1).long(), targets[:,t].long()]
            pose_t = positions[:,t]
            #obs_t = self.env_wrapper.step([demo_rgb[:,t], demo_depth[:,t], pose_t, target_goal, torch.ones(self.B).cuda()*t, (~masks).detach().cpu().numpy()])
            obs_t=demo_rgb[:,t]
            if t < lengths[0]:
                results['imgs'].append(demo_rgb[0,t].cpu().numpy())
                results['target'].append(target_goal[0].cpu().numpy())
                results['position'].append(positions[0,t].cpu().numpy())
                results['have_been'].append(aux_info['have_been'][0,t].cpu().numpy())
                results['distance'].append(aux_info['distance'][0,t].cpu().numpy())

            gt_act = copy.deepcopy(demo_act[:, t])
            if -100 in actions:
                b = torch.where(actions==-100)
                actions[b] = 0
            (
                pred_act,
                actions_logits,
            ) = self.agent(
                obs_t, target_goal
            )
            if not (gt_act == -100).all():
                loss = F.cross_entropy(actions_logits.view(-1,actions_logits.shape[1]),gt_act.long().view(-1))#, weight=action_weight)

                valid_indices = gt_act.long() != -100

                losses.append(loss)

            else:
                results['actions'].append(-1)
                results['gt_actions'].append(-1)

            actions = demo_act[:,t].contiguous()

        action_loss = torch.stack(losses).mean()

        total_loss = action_loss
        if train:
            self.optim.zero_grad()
            total_loss.backward(retain_graph=True)
            self.optim.step()

        loss_dict = {}
        loss_dict['loss'] = action_loss.item()

        return results, loss_dict


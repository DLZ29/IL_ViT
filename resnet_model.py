import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms

class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample()

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

class CategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return CustomFixedCategorical(logits=x), x

class ResNetForNavigation(nn.Module):
    def __init__(self):
        super(ResNetForNavigation, self).__init__()
        self.resize = transforms.Resize((224, 224))
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.load_state_dict(torch.load('resnet18_pretrained.pth'))
        self.resnet.fc = nn.Identity()
        self.linear = CategoricalNet(512*2, 4)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def preprocess(self, images, target_size=(224, 224)):
        images = images[:, :, :, :3]
        images = images.permute(0, 3, 1, 2)  # Convert to (N, C, H, W)
        transform = transforms.Compose([
            transforms.Resize(target_size),  # Resize to the target size
        ])
        transformed_images = [transform(self.to_pil(img)) for img in images]
        transformed_images = torch.stack([self.to_tensor(img) for img in transformed_images]).to(self.device)
        return transformed_images

    def forward(self, observation, target_goal):
        '''print("obsevation shape:", observation.shape)
        print("target_goal shape:", target_goal.shape)
        print("target_goal min:", target_goal.min())
        print("target_goal max:", target_goal.max())

        for i in range(4):
            print(f"Channel {i} min:", target_goal[:, :, :, i].min())
            print(f"Channel {i} max:", target_goal[:, :, :, i].max())'''
        observation = observation.to(self.device)
        target_goal = target_goal.to(self.device)
        observation = self.preprocess(observation)
        target_goal = self.preprocess(target_goal)
        observation = self.resnet(observation)
        target_goal = self.resnet(target_goal)
        combined = torch.cat((observation, target_goal), dim=1)
        distribution, x = self.linear(combined)
        action = distribution.sample()
        action_log_probs = distribution.log_probs(action)
        return action, x


'''model = ResNetForNavigation()

input_tensor1 = torch.randn(4, 144, 192, 3)
input_tensor2 = torch.randn(4, 144, 192, 3)

action, x = model(input_tensor1, input_tensor2)

print(f"Action shape: {action.shape}")
print(f"X shape: {x.shape}")
print(action)

torch.cuda.empty_cache()'''

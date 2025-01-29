import torch
from torch import nn
import timm
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
        self.fc1 = nn.Linear(num_inputs, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_outputs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return CustomFixedCategorical(logits=x), x

class CompressionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, grid_size, approx_output_size=1024):
        super(CompressionLayer, self).__init__()
        self.grid_size = grid_size
        num_patches = grid_size[0] * grid_size[1]
        num_channels = int(round(approx_output_size / num_patches))
        self.layer = nn.Sequential(
            nn.Conv2d(input_dim, num_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=num_channels),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

    def forward(self, x):
        batch_size, num_patches, dim = x.size()
        h, w = self.grid_size
        assert num_patches == h * w, f"Number of patches {num_patches} does not match grid size {h}x{w}"
        x = x.view(batch_size, h, w, dim).permute(0, 3, 1, 2)  # reshape to (batch_size, dim, h, w)
        x = self.layer(x)  # Apply compression layers
        return x

class OVRL(nn.Module):
    def __init__(self):
        super(OVRL, self).__init__()
        self.resize = transforms.Resize((224, 224))
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.flatten = nn.Flatten()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=False)
        self.vit.load_state_dict(torch.load('vit_small_patch16_224.pth', map_location='cuda:0'))
        for param in self.vit.parameters():
            param.requires_grad = False
        self.compression_layer = CompressionLayer(input_dim=384, output_dim=64, grid_size=(14, 14))
        self.linear = CategoricalNet(1960, 4)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def preprocess(self, images, target_size=(224, 224)):
        with torch.no_grad():
            images = images[:, :, :, :3]
            images = images.permute(0, 3, 1, 2)  # Convert to (N, C, H, W)

            # Define the transformation
            transform = transforms.Compose([
                transforms.Resize(target_size),  # Resize to the target size
                transforms.ToTensor()  # Convert back to tensor
            ])

            # Apply transformation to each image
            transformed_images = [transform(self.to_pil(img)) for img in images]

            # Stack the transformed images to form a single tensor
            transformed_images = torch.stack(transformed_images)

        return transformed_images

    def forward(self, observation, target_goal):
        observation = self.preprocess(observation)
        target_goal = self.preprocess(target_goal)
        observation = observation.to(self.device)
        target_goal = target_goal.to(self.device)

        observation = self.vit.forward_features(observation)
        target_goal = self.vit.forward_features(target_goal)

        observation = observation[:, 1:, :]
        target_goal = target_goal[:, 1:, :]

        observation = self.compression_layer(observation)
        target_goal = self.compression_layer(target_goal)

        combined = torch.cat((observation, target_goal), dim=1)
        combined = self.flatten(combined)
        distribution, x = self.linear(combined)
        action = distribution.sample()
        action_log_probs = distribution.log_probs(action)
        return action, x


'''
model = OVRL()

input_tensor1 = torch.randn(4, 144, 192, 3)
input_tensor2 = torch.randn(4, 144, 192, 3)

action, x = model(input_tensor1, input_tensor2)

print(f"Action shape: {action.shape}")
print(f"X shape: {x.shape}")
print(action)

torch.cuda.empty_cache()'''






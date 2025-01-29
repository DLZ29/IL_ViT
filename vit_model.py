import torch
from torch import nn
import timm
import torchvision.transforms as transforms

# 自定义分类网络，带有自定义采样和log_probs计算方法
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

# 定义分类网络
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

# 定义ViT模型用于导航任务
class ViTForNavigation(nn.Module):
    def __init__(self):
        super(ViTForNavigation, self).__init__()
        self.resize = transforms.Resize((224, 224))
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.Flatten = nn.Flatten()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=False)
        self.vit.load_state_dict(torch.load('vit_small_patch16_224.pth', map_location='cuda:0'))
        self.linear = CategoricalNet(2000, 4)  # 输入大小为2000
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def preprocess(self, images, target_size=(224, 224)):
        with torch.no_grad():
            images = images[:, :, :, :3]
            images = images.permute(0, 3, 1, 2)  # 转换为(N, C, H, W)

            # 定义转换
            transform = transforms.Compose([
                transforms.Resize(target_size),  # 调整到目标大小
                transforms.ToTensor()  # 转换回张量
            ])

            # 对每张图像应用转换
            transformed_images = [transform(self.to_pil(img)) for img in images]

            # 将转换后的图像堆叠成一个张量
            transformed_images = torch.stack(transformed_images)

        return transformed_images

    def forward(self, observation, target_goal):

        observation = self.preprocess(observation)
        target_goal = self.preprocess(target_goal)
        observation = observation.to(self.device)
        target_goal = target_goal.to(self.device)



        observation = self.vit(observation)
        target_goal = self.vit(target_goal)


        combined = torch.cat((observation, target_goal), dim=1)
        combined = self.Flatten(combined)
        distribution, x = self.linear(combined)
        action = distribution.sample()
        action_log_probs = distribution.log_probs(action)


        return action, x

    def print_memory_usage(self, phase=""):
        memory_allocated = torch.cuda.memory_allocated() / (1024**2)  # 转换为MB
        max_memory_allocated = torch.cuda.max_memory_allocated() / (1024**2)  # 转换为MB
        print(f"{phase} | 当前显存使用: {memory_allocated:.2f} MB | 最大显存使用: {max_memory_allocated:.2f} MB")




'''model = ViTForNavigation()

input_tensor1 = torch.randn(4, 144, 192, 3)
input_tensor2 = torch.randn(4, 144, 192, 4)

action, x = model(input_tensor1, input_tensor2)

print(f"Action shape: {action.shape}")
print(f"X shape: {x.shape}")
print(action)

torch.cuda.empty_cache()'''





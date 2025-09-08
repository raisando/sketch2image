
from abc import ABC, abstractmethod
import torch
from tqdm import tqdm
import probability_path as pp
import simple_model as model

class Trainer(ABC):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {idx}, loss: {loss.item()}')

        # Finish
        self.model.eval()

class ConditionalFlowMatchingTrainer(Trainer):
    def __init__(self, path: pp.ConditionalProbabilityPath, model: model.MLPVectorField, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size) # (bs, dim)
        t = torch.rand(batch_size,1).to(z) # (bs, 1)
        x = self.path.sample_conditional_path(z,t) # (bs, dim)

        ut_theta = self.model(x,t) # (bs, dim)
        ut_ref = self.path.conditional_vector_field(x,z,t) # (bs, dim)
        error = torch.sum(torch.square(ut_theta - ut_ref), dim=-1) # (bs,)
        return torch.mean(error)



class ConditionalScoreMatchingTrainer(Trainer):
    def __init__(self, path: pp.ConditionalProbabilityPath, model: model.MLPScore, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size) # (bs, dim)
        t = torch.rand(batch_size,1).to(z) # (bs, 1)
        x = self.path.sample_conditional_path(z,t) # (bs, dim)

        s_theta = self.model(x,t) # (bs, dim)
        s_ref = self.path.conditional_score(x,z,t) # (bs, dim)
        mse = torch.sum(torch.square(s_theta - s_ref), dim=-1) # (bs,)
        return torch.mean(mse)
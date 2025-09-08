import torch
import probability_path as pp
import sim_utils as sim
import simple_model as model
import alpha_beta as ab

class ConditionalVectorFieldODE(sim.ODE):
    def __init__(self, path: pp.ConditionalProbabilityPath, z: torch.Tensor):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, dim)
        """
        super().__init__()
        self.path = path
        self.z = z

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the conditional vector field u_t(x|z)
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        bs = x.shape[0]
        z = self.z.expand(bs, *self.z.shape[1:])
        return self.path.conditional_vector_field(x,z,t)


class LearnedVectorFieldODE(sim.ODE):
    def __init__(self, net: model.MLPVectorField):
        self.net = net

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (bs, dim)
            - t: (bs, dim)
        Returns:
            - u_t: (bs, dim)
        """
        return self.net(x, t)


class LangevinFlowSDE(sim.SDE):
    # def __init__(self, flow_model: model.MLPVectorField, score_model: model.MLPScore, sigma: float):
    #     """
    #     Args:
    #     - path: the ConditionalProbabilityPath object to which this vector field corresponds
    #     - z: the conditioning variable, (1, dim)
    #     """
    #     super().__init__()
    #     self.flow_model = flow_model
    #     self.score_model = score_model
    #     self.sigma = sigma

    def __init__(self, flow_model: model.MLPVectorField, sigma: float, alpha: ab.Alpha, beta: ab.Beta):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, dim)
        """
        super().__init__()
        self.flow_model = flow_model        
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta


    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        flow_value = self.flow_model(x,t)
        score_value = (self.alpha(t) * flow_value - self.alpha.dt(t) * x) 
        score_value /= (torch.square(self.beta(t)) * self.alpha.dt(t) - self.alpha(t)*self.beta.dt(t)*self.beta(t))
        return flow_value + 0.5 *self.sigma ** 2 * score_value

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.sigma * torch.randn_like(x)
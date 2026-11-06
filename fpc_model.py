import torch
import torch.nn as nn
from predictor import FlowPredictor, PredictionLoss
from corrector import PhysicsCorrector


class FPCModel(nn.Module):
    def __init__(self, rho=None, mu=None, use_physics=True, collocation_points=None,
                 coordinate_scale=None, field_scale=None, field_offset=None):
        super(FPCModel, self).__init__()

        self.predictor = FlowPredictor(in_channels=4, out_channels=4)
        # 物理参数由调用者填写；None 对应 YAML 中的 null，说明见 corrector.py。
        self.corrector = PhysicsCorrector(rho=rho, mu=mu,
            collocation_points=collocation_points, coordinate_scale=coordinate_scale,
            field_scale=field_scale, field_offset=field_offset)
        self.use_physics = use_physics

        self.data_loss_fn = PredictionLoss()

        self.grad_ema_data = 0.0
        self.grad_ema_pinn = 0.0

    def forward(self, x):
        return self.predictor(x)

    def compute_loss(self, predictions, targets, coords, ema_alpha=0.99):
        data_loss, mae = self.data_loss_fn(predictions, targets)

        if not self.use_physics:
            return data_loss, {
                'total_loss': data_loss.item(),
                'data_loss': data_loss.item(),
                'physics_loss': 0.0,
                'lambda_physics': 0.0,
                'mae': mae.item()
            }

        physics_loss = self.corrector(self.predictor, coords)

        if physics_loss.requires_grad:
            data_grads = torch.autograd.grad(
                data_loss, self.predictor.parameters(),
                retain_graph=True, allow_unused=True
            )
            physics_grads = torch.autograd.grad(
                physics_loss, self.predictor.parameters(),
                retain_graph=True, allow_unused=True
            )

            with torch.no_grad():
                data_grad_norm = sum(
                    g.abs().mean() for g in data_grads if g is not None
                )
                physics_grad_norm = sum(
                    g.abs().mean() for g in physics_grads if g is not None
                )

            if self.grad_ema_data == 0.0 and self.grad_ema_pinn == 0.0:
                self.grad_ema_data = data_grad_norm.item()
                self.grad_ema_pinn = physics_grad_norm.item()
            else:
                self.grad_ema_data = (
                    ema_alpha * self.grad_ema_data +
                    (1 - ema_alpha) * data_grad_norm.item()
                )
                self.grad_ema_pinn = (
                    ema_alpha * self.grad_ema_pinn +
                    (1 - ema_alpha) * physics_grad_norm.item()
                )

            lambda_physics = (
                self.grad_ema_data / (self.grad_ema_pinn + 1e-9)
                if self.grad_ema_pinn > 0 else 1.0
            )
        else:
            lambda_physics = 1.0

        total_loss = data_loss + lambda_physics * physics_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
            'lambda_physics': lambda_physics,
            'mae': mae.item()
        }

        return total_loss, loss_dict

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.predictor(x)


get_model = FPCModel
get_loss = PredictionLoss

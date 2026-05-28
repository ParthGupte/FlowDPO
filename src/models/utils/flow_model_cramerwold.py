from models.utils.flow_model import *

class FlowModelCW(FlowModelBase):
    def __init__(self, model_config, training_config):
        super().__init__(model_config, training_config)
    
    def training_step(self, batch, batch_idx):
        X_1, class_labels, X_0, t = batch
        Z_1 = self.encode_image(X_1)
        Z_t = (1 - t[:, None, None, None]) * X_0 + t[:, None, None, None] * Z_1
        dZ_t = Z_1 - X_0
        w = torch.rand_like(dZ_t)
        loss = self.loss_fn(self(Z_t,t,class_labels)-dZ_t,w)
        self.train_losses.append(loss.detach().cpu())
        return loss

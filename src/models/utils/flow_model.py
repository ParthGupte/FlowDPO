from abc import abstractmethod
import torch
import lightning as L
from diffusers import UNet2DModel 
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import wandb
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import inspect
from torchmetrics import AUROC



class FlowModelBase(L.LightningModule):
    def __init__(self,model_config,training_config):
        super().__init__()
        self.fid_metric = FrechetInceptionDistance(normalize=True)
        self.fid_metric.eval()
        kid_metric = KernelInceptionDistance(normalize=True, subset_size=50, subsets=50)
        lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True)
        ssim_metric = StructuralSimilarityIndexMeasure()
        self.metrics_dict = {
            "FID": self.fid_metric,
            "KID": kid_metric,
            "SSIM": ssim_metric,
            "LPIPS": lpips_metric,
        }
        
        self.set_hyperparameters(model_config,training_config)
    
    def set_hyperparameters(self,model_config,training_config):
        self.loss_fn = training_config.LOSS_FN_CLASS()
        self.learning_rate = training_config.LR
        self.image_size = training_config.IMAGE_SIZE
        self.solver_class_fwd = training_config.SOLVER_CLASS_FWD
        self.solver_steps = training_config.SOLVER_STEPS
        self.optimizer_class = training_config.OPTIMIZER_CLASS
        self.save_n_images = training_config.SAVE_N_IMAGES
    
    @abstractmethod
    def forward(self, Z_t, t, class_labels):
        pass

    @abstractmethod
    def encode_image(self,X_1):
        pass
        
    @abstractmethod
    def decode_image(self,Z_1):
        pass 

    def on_train_epoch_start(self):
        self.train_losses = []
    
    def training_step(self, batch, batch_idx):
        X_1, class_labels, X_0, t = batch
        Z_1 = self.encode_image(X_1)
        Z_t = (1 - t[:, None, None, None]) * X_0 + t[:, None, None, None] * Z_1
        dZ_t = Z_1 - X_0
        loss = self.loss_fn(self(Z_t,t,class_labels),dZ_t)
        self.train_losses.append(loss.detach().cpu())
        return loss
    
    def on_train_epoch_end(self):
        try:
            self.train_losses
        except:
            pass
        else:
            avg_loss = torch.stack(self.train_losses).mean()
        
        # Log the average loss
            self.log("train_epoch_avg_loss", avg_loss, prog_bar=True)
            self.train_losses.clear()
        torch.cuda.empty_cache()
    
    def on_validation_epoch_start(self):
        self.real_imgs_to_log = []
        self.gen_imgs_to_log = []
        for metric in self.metrics_dict.values():
            metric.to(self.device)
    
    def on_test_epoch_start(self):
        self.real_imgs_to_log = []
        self.gen_imgs_to_log = []
        for metric in self.metrics_dict.values():
            metric.to(self.device)

    @torch.no_grad()
    def validation_step(self,batch, batch_idx):
        X_1, class_labels, X_0, t = batch
        real_imgs = torch.clamp(X_1,0,1)
        gen_images = self.generate_image(X_0,class_labels,self.solver_steps)
        self.update_metrics(real_imgs,gen_images)
        self.log_images(real_imgs,batch_idx,"real_imgs",self.save_n_images)
        self.log_images(gen_images,batch_idx,"generated_imgs",self.save_n_images)
    
    @torch.no_grad()
    def test_step(self,batch, batch_idx):
        X_1, class_labels, X_0, t = batch
        real_imgs = torch.clamp((X_1+1)/2,0,1)
        gen_images = self.generate_image(X_0,class_labels,self.solver_steps)
        self.update_metrics(real_imgs,gen_images)
        self.log_images(real_imgs,batch_idx,"real_imgs",self.save_n_images)
        self.log_images(gen_images,batch_idx,"generated_imgs",self.save_n_images)

    def log_images(self,X_1_batch,batch_idx,name,n_images = 10):
        batch_size = X_1_batch.size()[0]
        n_reps = n_images//batch_size
        rem = n_images%batch_size
        if name == "real_imgs":
            images = self.real_imgs_to_log
        elif name == "generated_imgs":
            images = self.gen_imgs_to_log


        if batch_idx <= n_reps:
            if batch_idx < n_reps:
                images.extend([
                    wandb.Image(X_1_batch[i], caption=f"{name} {batch_idx*batch_size+i}")
                for i in range(batch_size)
                ])
            elif batch_idx == n_reps:
                images.extend([
                    wandb.Image(X_1_batch[i], caption=f"{name} {batch_idx*batch_size+i}")
                    for i in range(rem)
                ])
    
    def on_validation_epoch_end(self):
        self.logger.experiment.log({"real_imgs":self.real_imgs_to_log})
        self.logger.experiment.log({"generated_imgs":self.gen_imgs_to_log})
        self.gen_imgs_to_log.clear()
        metrics_dict = self.compute_metrics()
        self.log_metrics(metrics_dict)
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        self.logger.experiment.log({"real_imgs":self.real_imgs_to_log})
        self.logger.experiment.log({"generated_imgs":self.gen_imgs_to_log})
        self.gen_imgs_to_log.clear()
        metrics_dict = self.compute_metrics()
        self.log_metrics(metrics_dict)
        torch.cuda.empty_cache()
    
    @torch.no_grad()
    def generate_image(self,X_0, class_labels, n_steps, clamp = True):
        self.eval()
        solver = self.solver_class_fwd(self)
        Z_1 = solver.solve(X_0,class_labels,n_steps)
        X_1 = self.decode_image(Z_1)
        if clamp:
            return torch.clamp(X_1,0,1)
        else:
            return X_1
    
    @torch.no_grad()
    def solve_noise(self,X_1, class_labels, n_steps, clamp = True):
        self.eval()
        solver = self.solver_class_fwd(self)
        Z_0 = solver.solve_reverse(X_1,class_labels,n_steps)
        X_0 = self.decode_image(Z_0)
        if clamp:
            return torch.clamp(X_0,0,1)
        else:
            return X_0
        
    @torch.no_grad()
    def update_metrics(self,real_imgs,gen_imgs,fid_only = False):
        if fid_only:
            metric = self.metrics_dict["FID"]
            metric.update(real_imgs,real = True)
            metric.update(gen_imgs,real = False)
        else:
            for metric_name, metric in self.metrics_dict.items():
                if metric_name not in ["LPIPS","SSIM"]:
                    metric.update(real_imgs,real = True)
                    metric.update(gen_imgs,real = False)
                else:
                    metric.update(real_imgs,gen_imgs)
        
    @torch.no_grad()
    def compute_metrics(self,fid_only = False):
        metrics_values_dict = {}
        if fid_only:
            metric = self.metrics_dict["FID"]
            metrics_values_dict["FID"] = metric.compute()
            metric.reset()
        else: 
            for metric_name, metric in self.metrics_dict.items():
                if metric_name != "KID":
                    metrics_values_dict[metric_name] = metric.compute()
                else:
                    if sum([f.shape[0] for f in metric.real_features]) < metric.subset_size or sum([f.shape[0] for f in metric.fake_features]) < metric.subset_size:
                        print(len(metric.real_features),len(metric.fake_features),metric.subset_size)
                        print("Skipping KID computation (sanity check or too few samples).")
                    else:
                        metrics_values_dict[metric_name+"_mean"], metrics_values_dict[metric_name+"_std"] = metric.compute()
                metric.reset()
        return metrics_values_dict
    
    @torch.no_grad()
    def log_metrics(self,metrics_values_dict):
        self.log_dict(metrics_values_dict, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(),lr = self.learning_rate)
        return optimizer
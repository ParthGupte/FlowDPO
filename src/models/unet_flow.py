from models.utils.flow_model import *

class FlowModel(FlowModelBase):
    def __init__(self,unet: UNet2DModel,model_config,training_config):
        super().__init__(model_config,training_config)
        self.unet = unet
        self.set_hyperparameters(model_config,training_config)
    
    
    def forward(self, X_t,t,class_labels):
        # print("z_t:", X_t.shape)
        # print("t:", t.shape)
        # print("c:", class_labels.shape)
        # assert X_t.shape[0] == t.shape[0] == class_labels.shape[0], f"{X_t.shape}, {t.shape}, {class_labels.shape}"
        output = self.unet(
                sample=X_t,
                timestep=t,
                class_labels = class_labels,
            )
        return output.sample
    
    def encode_image(self,X_1):
        return X_1
    
    def decode_image(self,Z_1):
        return Z_1
    
    def update_metrics(self, real_imgs, gen_imgs, fid_only=True):
        imgs_shape = real_imgs.shape
        real_imgs = real_imgs.expand(imgs_shape[0],3,imgs_shape[2],imgs_shape[3])
        gen_imgs = gen_imgs.expand(imgs_shape[0],3,imgs_shape[2],imgs_shape[3])
        return super().update_metrics(real_imgs, gen_imgs, fid_only)

class FlowModelCats(FlowModelBase):
    def __init__(self,unet: UNet2DModel,model_config,training_config):
        super().__init__(model_config,training_config)
        self.unet = unet
        self.set_hyperparameters(model_config,training_config)
    
    
    def forward(self, X_t,t,class_labels):
        # print("z_t:", X_t.shape)
        # print("t:", t.shape)
        # print("c:", class_labels.shape)
        # assert X_t.shape[0] == t.shape[0] == class_labels.shape[0], f"{X_t.shape}, {t.shape}, {class_labels.shape}"
        output = self.unet(
                sample=X_t,
                timestep=t
            )
        return output.sample
    
    def encode_image(self,X_1):
        return X_1
    
    def decode_image(self,Z_1):
        return Z_1
    
    # def training_step(self, batch, batch_idx):
    #     X_1, class_labels, X_0, t = batch
    #     Z_1 = self.encode_image(X_1)
    #     Z_t = (1 - t[:, None, None, None]) * X_0 + t[:, None, None, None] * Z_1
    #     dZ_t = Z_1 - X_0
    #     loss = self.loss_fn(self(Z_t,t),dZ_t)
    #     self.train_losses.append(loss.detach().cpu())
    #     return loss
    
    # @torch.no_grad()
    # def validation_step(self,batch, batch_idx):
    #     X_1, class_labels, X_0, t = batch
    #     real_imgs = torch.clamp((X_1+1)/2,0,1)
    #     gen_images = self.generate_image(X_0,self.solver_steps)
    #     self.update_metrics(real_imgs,gen_images)
    #     self.log_images(real_imgs,batch_idx,"real_imgs",self.save_n_images)
    #     self.log_images(gen_images,batch_idx,"generated_imgs",self.save_n_images)
    
    # @torch.no_grad()
    # def test_step(self,batch, batch_idx):
    #     X_1, class_labels, X_0, t = batch
    #     real_imgs = torch.clamp((X_1+1)/2,0,1)
    #     gen_images = self.generate_image(X_0,self.solver_steps)
    #     self.update_metrics(real_imgs,gen_images)
    #     self.log_images(real_imgs,batch_idx,"real_imgs",self.save_n_images)
    #     self.log_images(gen_images,batch_idx,"generated_imgs",self.save_n_images)
    
    # @torch.no_grad()
    # def generate_image(self,X_0, n_steps, clamp = True):
    #     self.eval()
    #     solver = self.solver_class_fwd(self)
    #     Z_1 = solver.solve(X_0,n_steps)
    #     X_1 = self.decode_image(Z_1)
    #     if clamp:
    #         return torch.clamp(X_1,0,1)
    #     else:
    #         return X_1
    
    # @torch.no_grad()
    # def solve_noise(self,X_1, n_steps, clamp = True):
    #     self.eval()
    #     solver = self.solver_class_fwd(self)
    #     Z_0 = solver.solve_reverse(X_1,n_steps)
    #     X_0 = self.decode_image(Z_0)
    #     if clamp:
    #         return torch.clamp(X_0,0,1)
    #     else:
    #         return X_0
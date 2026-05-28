from models.utils.flow_model_cramerwold import *

class FlowModel(FlowModelCW):
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
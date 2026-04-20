from models.utils.flow_model import *

class FlowDPOModule(FlowModelBase):
    def __init__(self, unet, unet_ref, model_config, training_config):
        super().__init__(model_config, training_config)

        self.unet = unet
        self.unet_ref = unet_ref

        # Freeze reference model
        for p in self.unet_ref.parameters():
            p.requires_grad = False

        self.unet_ref.eval()

        self.set_hyperparameters(model_config, training_config)
    
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
    
    def ref_forward(self, X_t,t,class_labels):
        # print("z_t:", X_t.shape)
        # print("t:", t.shape)
        # print("c:", class_labels.shape)
        # assert X_t.shape[0] == t.shape[0] == class_labels.shape[0], f"{X_t.shape}, {t.shape}, {class_labels.shape}"
        output = self.unet_ref(
                sample=X_t,
                timestep=t,
                class_labels = class_labels,
            )
        return output.sample
    
    def training_step(self, batch, batch_idx):
        y_t_til = lambda y_0, y_1, t: (1 - t[:, None, None, None]) * y_0 + t[:, None, None, None] * y_1
        delta_t_til = lambda y_t, t, x: self(y_t,t,x) - self.ref_forward(y_t,t,x)
        
        y_w = batch['y_pos']
        y0_w = batch['y0_pos']
        y_l = batch['y_neg']
        y0_l = batch['y0_neg']
        x = batch['x']
        t1 = batch['t1']
        t2 = batch['t2']
        t3 = batch['t3']
        t4 = batch['t4']
        t5 = batch['t5']
        eps = batch['eps']

        # ---- y_t outputs (winner branch) ----
        y_t1_w = y_t_til(y0_w, y_w, t1)
        y_t2_w = y_t_til(y0_w, y_w, t2)
        y_t5_w = y_t_til(y0_w, y_w, t5)

        # ---- y_t outputs (loser branch) ----
        y_t3_l = y_t_til(y0_l, y_l, t3)
        y_t4_l = y_t_til(y0_l, y_l, t4)
        y_t5_l = y_t_til(y0_l, y_l, t5)

        # ---- delta evaluations ----
        delta_t1_w = delta_t_til(y_t1_w, t1, x)
        delta_t2_w = delta_t_til(y_t2_w, t2, x)
        delta_t5_w = delta_t_til(y_t5_w, t5, x)

        delta_t3_l = delta_t_til(y_t3_l, t3, x)
        delta_t4_l = delta_t_til(y_t4_l, t4, x)
        delta_t5_l = delta_t_til(y_t5_l, t5, x)

        loss = self.loss_fn(delta_t1_w,delta_t2_w,delta_t3_l,delta_t4_l,delta_t5_l,delta_t5_w,y_w,y_l,eps,y_t5_l,y_t5_w)

        return loss

        


    def encode_image(self,X_1):
        return X_1
    
    def decode_image(self,Z_1):
        return Z_1
    
    def update_metrics(self, real_imgs, gen_imgs, fid_only=True):
        imgs_shape = real_imgs.shape
        real_imgs = real_imgs.expand(imgs_shape[0],3,imgs_shape[2],imgs_shape[3])
        gen_imgs = gen_imgs.expand(imgs_shape[0],3,imgs_shape[2],imgs_shape[3])
        return super().update_metrics(real_imgs, gen_imgs, fid_only)
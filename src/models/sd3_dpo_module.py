from models.utils.flow_model import FlowModelBase
from diffusers import StableDiffusion3Pipeline
import torch

class FlowDPOSD3(FlowModelBase):
    def __init__(self,pipe:StableDiffusion3Pipeline,ref_transformer, model_config, training_config):
        super().__init__(model_config, training_config)
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.ref_transformer = ref_transformer.eval()

    
    def encode_image(self, X):
        latent = self.pipe.vae.encode(
            X,
            return_dict=False
        )[0].sample()

        latent = (
            latent
            - self.pipe.vae.config.shift_factor
        ) * self.pipe.vae.config.scaling_factor

        return latent
    
    def decode_image(self, Z_1):
        Z_1 = (Z_1 / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        return self.pipe.vae.decode(Z_1, return_dict=False)[0]
    
    def encode_prompt(self,prompt: str | list[str],):
        prompt_embeds,_,pooled_prompt_embeds, _ = self.pipe.encode_prompt(prompt=prompt,prompt_2=None,prompt_3=None)
        cond = (prompt_embeds,pooled_prompt_embeds)
        return cond
    
    def forward(self, latent_model_input, timestep, cond):
        prompt_embeds,pooled_prompt_embeds = cond
        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            # joint_attention_kwargs=self.pipe.joint_attention_kwargs,
            return_dict=False,
        )[0]

        return noise_pred
    
    def ref_forward(self, latent_model_input, timestep, prompt_embeds,pooled_prompt_embeds):
        noise_pred = self.ref_transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=self.pipe.joint_attention_kwargs,
            return_dict=False,
        )[0]

        return noise_pred
    
    def training_step(self, batch, batch_idx):
        y_t_til = lambda y_0, y_1, t: (1 - t[:, None, None, None]) * y_0 + t[:, None, None, None] * y_1
        delta_t_til = lambda y_t, t, prompt_embeds,pooled_prompt_embeds: self(y_t,t,prompt_embeds,pooled_prompt_embeds) - self.ref_forward(y_t,t,prompt_embeds,pooled_prompt_embeds)

        y_w = batch['y_pos'] #pre embeded image
        y0_w = batch['y0_pos'] #pre computed noise
        y_l = batch['y_neg'] #pre embeded image
        y0_l = batch['y0_neg'] #pre computed noise
        cond = batch['prompt_embeds'] #pre embedded prompt
        t1 = batch['t1']
        t2 = batch['t2']
        t3 = batch['t3']
        t4 = batch['t4']
        t5 = batch['t5']



        # ---- y_t outputs (winner branch) ----
        y_t1_w = y_t_til(y0_w, y_w, t1)
        y_t2_w = y_t_til(y0_w, y_w, t2)
        y_t5_w = y_t_til(y0_w, y_w, t5)

        # ---- y_t outputs (loser branch) ----
        y_t3_l = y_t_til(y0_l, y_l, t3)
        y_t4_l = y_t_til(y0_l, y_l, t4)
        y_t5_l = y_t_til(y0_l, y_l, t5)

        y_t5_l = y_t5_l.detach().requires_grad_(True)
        y_t5_w = y_t5_w.detach().requires_grad_(True)

        # ---- delta evaluations ----
        delta_t1_w = delta_t_til(y_t1_w, t1, cond)
        delta_t2_w = delta_t_til(y_t2_w, t2, cond)
        delta_t5_w = delta_t_til(y_t5_w, t5, cond)

        delta_t3_l = delta_t_til(y_t3_l, t3, cond)
        delta_t4_l = delta_t_til(y_t4_l, t4, cond)
        delta_t5_l = delta_t_til(y_t5_l, t5, cond)

        loss = self.loss_fn(delta_t1_w,delta_t2_w,delta_t3_l,delta_t4_l,delta_t5_l,delta_t5_w,y_w,y_l,y_t5_l,y_t5_w)

        self.train_losses.append(loss.detach().cpu())
        
        return loss
    
    @torch.no_grad()
    def validation_step(self,batch, batch_idx):
        X_1 = batch['y_pos']
        prompt_embeds, pooled_prompt_embeds = batch['prompt_embeds']
        real_imgs = X_1
        gen_images_latents = self.pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            output_type='latents'
        ).images
        gen_images = self.decode_image(gen_images_latents)
        self.update_metrics(real_imgs,gen_images)
        self.log_images(real_imgs,batch_idx,"real_imgs",self.save_n_images)
        self.log_images(gen_images,batch_idx,"generated_imgs",self.save_n_images)
    
    @torch.no_grad()
    def generate_image(self,X_0, cond):
        self.eval()
        prompt_embeds, pooled_prompt_embeds = cond
        gen_images_latents = self.pipe(
            latents = X_0,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            output_type='latent'
        ).images
        # print(gen_images_latents)
        return gen_images_latents

    @torch.no_grad()
    def solve_noise(self,y_1,cond,num_inference_steps=28):
        self.eval()
        prompt_embeds, pooled_prompt_embeds = cond
        z = y_1.clone()
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(num_inference_steps,device=z.device)
        timesteps = scheduler.timesteps.flip(0)
        sigmas = scheduler.sigmas.flip(0)

        for i in range(len(sigmas)-1):
            t = timesteps[i]
            sigma = sigmas[i]
            sigma_next = sigmas[i+1]
            dt = sigma_next - sigma
            pred_v = self.forward(z,t.expand(z.shape[0]),(prompt_embeds, pooled_prompt_embeds))
            z = z + dt * pred_v
        return z
    
    @torch.no_grad()
    def minimal_generate(self,prompt,X_0 = None):
        cond = self.encode_prompt(prompt)
        BATCH_SIZE = cond[0].shape[0]
        IN_CHANNELS = self.transformer.config.in_channels
        if X_0 is None:
            X_0 = self.pipe.prepare_latents(BATCH_SIZE,IN_CHANNELS,1024,1024,cond[0].dtype,cond[0].device,None)
        gen_image_latents = self.generate_image(X_0,cond)
        final_img = self.decode_image(gen_image_latents).detach()
        final_img_pil = self.pipe.image_processor.postprocess(final_img)[0]
        return final_img_pil
        
        
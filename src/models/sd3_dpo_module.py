from models.utils.flow_model import FlowModelBase
from diffusers import StableDiffusion3Pipeline
from metrics.hpsv3.metric import HPSv3Metric
from peft import LoraConfig
import torch

class FlowDPOSD3(FlowModelBase):
    def __init__(self,pipe:StableDiffusion3Pipeline,ref_transformer, model_config, training_config):
        super().__init__(model_config, training_config)
        self.pipe = pipe
        # pipe.__call__ (used in validation_step/generate_image) spawns its own
        # tqdm bar per denoising loop, which spams the terminal on top of
        # Lightning's own progress bar across every validation batch.
        self.pipe.set_progress_bar_config(disable=True)
        self.transformer = pipe.transformer
        self.ref_transformer = ref_transformer.eval()
        self.ref_transformer.requires_grad_(False)

        self.use_lora = getattr(training_config, "USE_LORA", False)
        if self.use_lora:
            # freezes every non-adapter param on self.transformer automatically
            # (see peft.tuners.lora); ref_transformer never gets an adapter --
            # it must stay a plain frozen copy of the base weights for the DPO
            # delta comparison to be against the original model, not the LoRA one.
            lora_config = LoraConfig(
                r=training_config.LORA_RANK,
                lora_alpha=training_config.LORA_ALPHA,
                lora_dropout=training_config.LORA_DROPOUT,
                target_modules=training_config.LORA_TARGET_MODULES,
            )
            self.transformer.add_adapter(lora_config)

        # training_step runs 6 forward/backward passes per step (double-backward
        # for the trace-estimator term) under the math attention backend, which
        # OOMed even at batch_size=2 with ~90GB already resident before the
        # final allocation. Gradient checkpointing trades recompute for not
        # storing per-layer activations, which is what actually scales with
        # attention's O(seq_len^2) memory at 1024x1024 -- reducing batch_size
        # further wasn't closing the gap proportionally. ref_transformer also
        # needs it since its graph is retained for the t5 trace-estimator term.
        self.transformer.enable_gradient_checkpointing()
        self.ref_transformer.enable_gradient_checkpointing()
        # ref_transformer, fid_metric, etc. are reconstructed on every load
        # (load_sd3_model deepcopies the pretrained transformer), so only the
        # trainable transformer needs to be checkpointed.
        self.strict_loading = False
        self.hpsv3_metric = HPSv3Metric(device="cuda")

    def setup(self, stage=None):
        super().setup(stage)
        # batch['cond'] carries pre-embedded *positive* prompts (see
        # precompute_noise_hpdv3.py), but self.pipe(...) in validation_step
        # still needs a negative/empty-prompt embedding internally for
        # classifier-free guidance -- it's the same fixed embedding for every
        # batch, so compute it once now while the text encoders are still on
        # GPU and cache it, rather than leaving the encoders resident on GPU
        # (~13GB, needed for the double-backward computation in training_step,
        # which was OOMing at batch_size=5 even after VAL_BATCH_SIZE was
        # already split off from BATCH_SIZE) just to redo this every call.
        if not hasattr(self, "neg_prompt_embeds"):
            _, neg_prompt_embeds, _, neg_pooled_prompt_embeds = self.pipe.encode_prompt(
                prompt=[""], prompt_2=None, prompt_3=None, do_classifier_free_guidance=True,
            )
            self.register_buffer("neg_prompt_embeds", neg_prompt_embeds, persistent=False)
            self.register_buffer("neg_pooled_prompt_embeds", neg_pooled_prompt_embeds, persistent=False)

        # text_encoder (CLIP-L, ~0.25GB) is deliberately left on GPU:
        # DiffusionPipeline.device/_execution_device pick the device of the
        # first nn.Module in alphabetically-sorted component names, and
        # "text_encoder" sorts before "transformer" -- moving it off GPU
        # silently makes self.pipe.device report "cpu", so pipe(...) then
        # creates latents/timesteps on CPU while self.transformer stays on
        # GPU, crashing with a device-mismatch. text_encoder_2 (~1.4GB) and
        # text_encoder_3 (T5-XXL, ~11.5GB) are the ones actually worth freeing.
        self.pipe.text_encoder_2.to("cpu")
        self.pipe.text_encoder_3.to("cpu")

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        # hpsv3_metric (~16.6GB) is only used in validation_step; keeping it on
        # GPU during training steals memory training's double-backward pass needs.
        self.hpsv3_metric.to("cpu")
        torch.cuda.empty_cache()

    def train(self, mode: bool = True):
        # ref_transformer is a registered submodule, so Lightning's automatic
        # model.train() at the start of each training epoch would otherwise
        # pull it out of eval mode along with the trainable transformer, even
        # though requires_grad_(False) alone doesn't stop that. It must stay
        # frozen and deterministic for the DPO delta comparison to be valid.
        super().train(mode)
        self.ref_transformer.eval()
        return self

    def on_save_checkpoint(self, checkpoint):
        if self.use_lora:
            # base transformer weights are unmodified pretrained weights (peft
            # freezes them), so only the LoRA adapter matrices are worth saving.
            checkpoint["state_dict"] = {
                k: v for k, v in checkpoint["state_dict"].items()
                if k.startswith("transformer.") and "lora_" in k
            }
        else:
            checkpoint["state_dict"] = {
                k: v for k, v in checkpoint["state_dict"].items()
                if k.startswith("transformer.")
            }


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
    
    def ref_forward(self, latent_model_input, timestep, cond):
        prompt_embeds,pooled_prompt_embeds = cond
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
        delta_t_til = lambda y_t, t, cond: self(y_t,t,cond) - self.ref_forward(y_t,t,cond)
        def delta_t_til_frozen_ref(y_t, t, cond):
            # Only the t5 deltas feed torch.autograd.grad(create_graph=True) below
            # (the trace-estimator term), which needs gradients through the
            # ref_transformer path too (d(delta)/d(y_t), even though its weights
            # are frozen). The t1-t4 deltas only ever need gradients into
            # self.transformer's weights via loss_fn's term_1/term_2, for which
            # ref_forward's activation graph is dead weight -- retaining it
            # anyway across all 6 forward passes accumulated enough memory to
            # OOM training_step regardless of batch_size.
            with torch.no_grad():
                ref = self.ref_forward(y_t, t, cond)
            return self(y_t, t, cond) - ref

        y_w = batch['y_pos'] #pre embeded image
        y0_w = batch['y0_pos'] #pre computed noise
        y_l = batch['y_neg'] #pre embeded image
        y0_l = batch['y0_neg'] #pre computed noise
        cond = batch['cond'] #pre embedded prompt
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
        delta_t1_w = delta_t_til_frozen_ref(y_t1_w, t1, cond)
        delta_t2_w = delta_t_til_frozen_ref(y_t2_w, t2, cond)
        delta_t5_w = delta_t_til(y_t5_w, t5, cond)

        delta_t3_l = delta_t_til_frozen_ref(y_t3_l, t3, cond)
        delta_t4_l = delta_t_til_frozen_ref(y_t4_l, t4, cond)
        delta_t5_l = delta_t_til(y_t5_l, t5, cond)

        loss = self.loss_fn(delta_t1_w,delta_t2_w,delta_t3_l,delta_t4_l,delta_t5_l,delta_t5_w,y_w,y_l,y_t5_l,y_t5_w)

        self.train_losses.append(loss.detach().cpu())
        
        return loss
    
    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        # undo the on_train_epoch_start offload; also covers the standalone
        # trainer.validate() baseline call, where it's already on GPU from __init__.
        self.hpsv3_metric.to(self.device)
        self.hpsv3_scores = []

    @torch.no_grad()
    def validation_step(self,batch, batch_idx):
        X_1 = batch['y_pos']
        prompts = batch['prompt']
        prompt_embeds, pooled_prompt_embeds = batch['cond']
        real_imgs = torch.clamp((self.decode_image(X_1) + 1) / 2, 0, 1)
        batch_size = prompt_embeds.shape[0]
        gen_images_latents = self.pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=self.neg_prompt_embeds.expand(batch_size, -1, -1),
            negative_pooled_prompt_embeds=self.neg_pooled_prompt_embeds.expand(batch_size, -1),
            output_type='latent'
        ).images
        gen_images = torch.clamp((self.decode_image(gen_images_latents) + 1) / 2, 0, 1)
        self.update_metrics(real_imgs,gen_images)
        self.hpsv3_scores.append(self.hpsv3_metric.score(prompts, gen_images))
        self.log_images(real_imgs,batch_idx,"real_imgs",self.save_n_images)
        self.log_images(gen_images,batch_idx,"generated_imgs",self.save_n_images)

    def on_validation_epoch_end(self):
        self.log("HPSv3", torch.cat(self.hpsv3_scores).mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.hpsv3_scores.clear()
        super().on_validation_epoch_end()
    
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
    

class FLowDPOSD3Wallace(FlowDPOSD3):
    def training_step(self, batch, batch_idx):
        y_t_til = lambda y_0, y_1, t: (1 - t[:, None, None, None]) * y_0 + t[:, None, None, None] * y_1
        V_ideal = lambda y_0, y_1: (y_1 - y_0)

        y_w = batch['y_pos']
        y0_w = torch.randn_like(y_w)
        y_l = batch['y_neg']
        y0_l = torch.randn_like(y_l)
        cond = batch['cond']
        t1 = batch['t1']
        t2 = batch['t2']
        t3 = batch['t3']
        t4 = batch['t4']
        t5 = batch['t5']

        V_w = V_ideal(y0_w, y_w)
        V_l = V_ideal(y0_l, y_l)

        # ---- y_t outputs (winner branch) ----
        y_t_w = y_t_til(y0_w, y_w, t1)
        y_t_l = y_t_til(y0_l, y_l, t1)

        V_theta = self(y_t_w,t1,cond)
        V_ref = self.ref_forward(y_t_l,t1,cond)

        loss = self.loss_fn(V_theta,V_ref,V_w,V_l)

        self.train_losses.append(loss.detach().cpu())
        
        return loss
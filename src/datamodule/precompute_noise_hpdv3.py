import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from HPDv3_dataloader import HPDv3Dataset
from experiments.configs import training_config_3_dpo as training_config, model_config_1 as model_config
from inference_tests.utils.load_sd3_model import load_sd3_model
from models.sd3_dpo_module import FlowDPOSD3 
import os


def _atomic_torch_save(obj, path):
    """Write to a temp file then rename, so save_path is never left as a
    partially-written file if the process dies mid-write."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def compute_x0_dataset(model, dataloader, save_path="data/hpdv3_pairs.pt", save_every=50):
    meta_path = save_path + ".meta"

    if os.path.exists(save_path):
        results = torch.load(save_path, map_location="cpu")
        meta = torch.load(meta_path)
        start_batch = meta["last_batch"] + 1
        print(f"Loaded {len(results)} examples")
        print(f"Resuming from batch {start_batch}")
    else:
        results = []
        start_batch = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if batch_idx < start_batch:
                continue

            y_w = batch["winner"].to(model.device)
            y_l = batch["loser"].to(model.device)
            prompts = batch["prompt"]

            y_w_enc = model.encode_image(y_w)
            y_l_enc = model.encode_image(y_l)
            cond = model.encode_prompt(prompts)

            x0_w = model.solve_noise(y_w_enc, cond, model.solver_steps)
            x0_l = model.solve_noise(y_l_enc, cond, model.solver_steps)

            for i in range(y_w.shape[0]):
                results.append({
                    "y_w_enc": y_w_enc[i].half().cpu(),
                    "y_l_enc": y_l_enc[i].half().cpu(),
                    "x0_w": x0_w[i].half().cpu(),
                    "x0_l": x0_l[i].half().cpu(),
                    "prompt_embeds": cond[0][i].half().cpu(),
                    "pooled_prompt_embeds": cond[1][i].half().cpu(),
                    "prompt": prompts[i],
                })

            if (batch_idx + 1) % save_every == 0:
                _atomic_torch_save(results, save_path)
                _atomic_torch_save({"last_batch": batch_idx}, meta_path)
                print(f"Saved checkpoint at batch {batch_idx}")

    _atomic_torch_save(results, save_path)
    _atomic_torch_save({"last_batch": batch_idx}, meta_path)
    print("Finished.")
    print(f"Saved {len(results)} examples")

    return results



if __name__ == "__main__":
    from models.sd3 import pipe
    flow_model = load_sd3_model(pipe,model_config,training_config)

    dataset_train = HPDv3Dataset(split = 'train')
    dataloader_train = DataLoader(dataset_train,training_config.BATCH_SIZE,shuffle=False)

    data = compute_x0_dataset(flow_model,dataloader_train)
    print(data[0])

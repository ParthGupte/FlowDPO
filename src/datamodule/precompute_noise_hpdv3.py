import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from HPDv3_dataloader import HPDv3Dataset
from experiments.configs import training_config_3_dpo as training_config, model_config_1 as model_config
from inference_tests.utils.load_sd3_model import load_sd3_model
from models.sd3_dpo_module import FlowDPOSD3 

def compute_x0_dataset(model:FlowDPOSD3, dataloader:DataLoader):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            y_w, y_l = batch['winner'].to(model.device), batch['loser'].to(model.device)
            prompts = batch['prompt']

            y_w_enc = model.encode_image(y_w)
            y_l_enc = model.encode_image(y_l)

            cond = model.encode_prompt(prompts)

            x0_w = model.solve_noise(y_w_enc,cond,model.solver_steps)   
            x0_l = model.solve_noise(y_l_enc,cond,model.solver_steps)
            for i in range(y_w.shape[0]):
                results.append({
                    "y_w_enc": y_w_enc[i].cpu(),
                    "y_l_enc": y_l_enc[i].cpu(),
                    "x0_w": x0_w[i].cpu(),
                    "x0_l": x0_l[i].cpu(),
                    "cond": (cond[0][i].cpu(),cond[1][i].cpu())
                })

    return results



if __name__ == "__main__":
    from models.sd3 import pipe
    flow_model = load_sd3_model(pipe,model_config,training_config)

    dataset_train = HPDv3Dataset(split = 'train')
    dataloader_train = DataLoader(dataset_train,training_config.BATCH_SIZE,shuffle=False)

    data = compute_x0_dataset(flow_model,dataloader_train)
    print(data[0])

    torch.save(data,"data/hpdv3_pairs.pt")

import torch
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from datamodule.MNIST_loader import get_mnist_dataset
from models.unet_flow import FlowModel, UNet2DModel
from experiments.configs import training_config_1 as training_config, model_config_1 as model_config
from inference_tests.utils.load_model import load_model_mnist

flip = transforms.RandomHorizontalFlip(p=1.0)

def compute_x0_dataset(model, dataloader, digit_filter = [9]):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            x1, labels = batch

            # filter only 9s
            mask = torch.isin(labels, torch.tensor(digit_filter, device=labels.device))
            x1 = x1[mask].to(model.device)
            labels = labels[mask].to(model.device)
            if x1.shape[0] == 0:
                continue
    
            x0 = model.solve_noise(x1,labels,model.solver_steps)   
            x1_flip = flip(x1)
            x0_flip = model.solve_noise(x1_flip,labels,model.solver_steps) 
            for i in range(x1.shape[0]):
                results.append({
                    "x1": x1[i].cpu(),
                    "x1_flip": x1_flip[i].cpu(),
                    "x0": x0[i].cpu(),
                    "x0_flip": x0_flip[i].cpu(),
                    "label": labels[i].cpu()
                })

    return results



if __name__ == "__main__":
    ckpt_path = "flow_dpo_training_logs/gdx9e3sy/checkpoints/FlowModel-epoch=94-FID=14.2297.ckpt"
    flow_model = load_model_mnist(ckpt_path,model_config,training_config)

    dataset_train = get_mnist_dataset(train = True)
    dataloader_train = DataLoader(dataset_train,training_config.BATCH_SIZE,shuffle=False)

    data = compute_x0_dataset(flow_model,dataloader_train,digit_filter=[9])
    print(data[0])

    torch.save(data,"data/mnist_9_flow_pairs.pt")

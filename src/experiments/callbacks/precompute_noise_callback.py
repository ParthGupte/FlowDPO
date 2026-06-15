import os
import glob
import torch
import lightning as L
from torch.utils.data import DataLoader

from datamodule.MNIST_loader import get_mnist_dataset
from datamodule.precompute_noise import compute_x0_dataset
from inference_tests.utils.load_model import load_flowmodel_from_dpo_ckpt


class PrecomputeNoiseCallback(L.Callback):
    def __init__(self, datamodule, model_config, training_config, interval_epochs=5, output_dir="data", digit_filter=[9]):
        super().__init__()
        self.datamodule = datamodule
        self.model_config = model_config
        self.training_config = training_config
        self.interval_epochs = interval_epochs
        self.output_dir = output_dir
        self.digit_filter = digit_filter

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at end of each training epoch (more reliable than on_epoch_end)"""
        import sys
        try:
            epoch = int(trainer.current_epoch)
        except Exception:
            epoch = 0

        # Debug: print every epoch to verify callback is being called
        print(f"[DEBUG] PrecomputeNoiseCallback.on_train_epoch_end called at epoch {epoch}, interval={self.interval_epochs}", flush=True)
        sys.stdout.flush()

        if (epoch + 1) % self.interval_epochs != 0:
            print(f"[DEBUG] Skipping precompute at epoch {epoch} (not a multiple of {self.interval_epochs})", flush=True)
            sys.stdout.flush()
            return

        print(f"PrecomputeNoiseCallback: starting precompute at epoch {epoch+1}", flush=True)
        sys.stdout.flush()

        # find latest checkpoint file
        ckpts = glob.glob("flow_dpo_training_logs/**/checkpoints/*.ckpt", recursive=True)
        if len(ckpts) == 0:
            print("PrecomputeNoiseCallback: no checkpoints found, skipping", flush=True)
            sys.stdout.flush()
            return

        latest_ckpt = max(ckpts, key=os.path.getmtime)
        print(f"PrecomputeNoiseCallback: using checkpoint {latest_ckpt}", flush=True)
        sys.stdout.flush()

        try:
            # Load FlowModel built from the latest DPO checkpoint
            print("PrecomputeNoiseCallback: loading FlowModel from DPO checkpoint...", flush=True)
            flow_model = load_flowmodel_from_dpo_ckpt(latest_ckpt, self.model_config, self.training_config)

            # ensure model is on the same device as the training module
            try:
                target_device = pl_module.device
            except Exception:
                target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            flow_model.to(target_device)
            flow_model.eval()

            # build a simple MNIST dataloader for precompute
            dataset = get_mnist_dataset(train=True)
            dl = DataLoader(dataset, batch_size=self.training_config.BATCH_SIZE, shuffle=False, num_workers=0)

            # compute
            data = compute_x0_dataset(flow_model, dl, digit_filter=self.digit_filter)

            # save with epoch in name
            os.makedirs(self.output_dir, exist_ok=True)
            save_path = os.path.join(self.output_dir, f"mnist_9_flow_pairs_epoch{epoch+1}.pt")
            torch.save(data, save_path)
            print(f"PrecomputeNoiseCallback: saved precomputed data to {save_path}", flush=True)
            sys.stdout.flush()

            # update datamodule to use new file and re-setup
            print("PrecomputeNoiseCallback: updating datamodule...", flush=True)
            sys.stdout.flush()
            
            self.datamodule.data_path = save_path
            self.datamodule.setup(stage="fit")
            
            # Reattach the datamodule/dataloaders through Lightning's internal connector
            trainer._data_connector.attach_datamodule(pl_module, self.datamodule)
            
            print("PrecomputeNoiseCallback: datamodule updated to new precomputed data", flush=True)
            sys.stdout.flush()
        except Exception as e:
            import traceback
            print(f"PrecomputeNoiseCallback: error during precompute/update: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()

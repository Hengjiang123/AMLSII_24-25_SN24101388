# !pip install efficientnet_pytorch
import sys, os, random, numpy as np, torch, pandas as pd
sys.path.append('../input')

from A.datasets  import load_data, stratified_split, make_balanced_augmented
from A.evaluate  import cross_validate
from A.test      import evaluate_test

# from amls2final7.amls2folder.A.datasets  import load_data, stratified_split, make_balanced_augmented
# from amls2final7.amls2folder.A.evaluate  import cross_validate
# from amls2final7.amls2folder.A.test      import evaluate_test

class MainConfig:
    """
    Central config
    ──────────────
    - dataset_mode        : "original" | "balanced_aug"
    - lr_scheduler_type   : "plateau"  | "cosine"
    - backbone            : "efficientnet-b2" | "efficientnet-b3" | "efficientnet-b4"
    """
    def __init__(self):
        # environment
        if "KAGGLE_URL_BASE" in os.environ:
            self.environment = "kaggle"
        elif "COLAB_GPU" in os.environ:
            self.environment = "colab"
        else:
            self.environment = "local"

        # paths
        if self.environment == "kaggle":
            self.data_dir   = "/kaggle/input/jpeg-melanoma-384x384/train/"
            self.csv_path   = "/kaggle/input/amls2final7/amls2folder/Datasets/train.csv"
            self.tensor_dir = "/kaggle/input/siimtensor384/preprocessed_384"
            self.weights_dir = "/kaggle/working/weights/"
            self.images_dir  = "/kaggle/working/images/"
        else:                              # local or colab
            self.data_dir   = "datasets/train/"
            self.csv_path   = "datasets/train.csv"
            self.tensor_dir = "datasets/preprocessed_384/"
            self.weights_dir = "weights/"
            self.images_dir  = "images/"

        # parameters
        self.num_classes     = 2
        self.img_size        = 384
        self.batch_size      = 16
        self.num_workers     = 4
        self.epochs          = 40
        self.learning_rate   = 1e-4
        self.weight_decay    = 1e-4       
        self.patience        = 6
        self.random_seed     = 42
        self.num_folds       = 3

        # switches
        self.use_preprocessed   = False
        self.use_meta           = True
        self.use_focal_loss     = False
        self.use_class_weights  = True
        self.label_smoothing    = 0.1    
        self.classifier_dropout = 0.5     
        self.use_sampler        = True
        self.use_scheduler      = True
        self.lr_scheduler_type  = "cosine"   # <-- "plateau" or "cosine"
        self.warmup_epochs      = 5          # warm‑up steps for cosine
        self.cosine_epochs = self.epochs - self.warmup_epochs
        self.dataset_mode       = "original"   # "balanced_aug" or "original"
        self.test_mode          = False
        self.use_tqdm           = True

        # Backbone 
        self.backbone = "efficientnet-b4"     # "efficientnet-b2"|"efficientnet-b3"|"efficientnet-b4"

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_meta_features = 0  

def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# -------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = MainConfig()
    _set_seed(cfg.random_seed)
    os.makedirs(cfg.weights_dir, exist_ok=True); os.makedirs(cfg.images_dir, exist_ok=True)

    print(f"Env={cfg.environment}  dataset={cfg.dataset_mode}  backbone={cfg.backbone}")
    print(f"LR‑sched={cfg.lr_scheduler_type}  label_smooth={cfg.label_smoothing}\n")

    # Data loading
    df = load_data(cfg)
    if cfg.dataset_mode == "balanced_aug":
        df = make_balanced_augmented(df, random_state=cfg.random_seed)
        print(f"[Balanced/Aug] size={len(df)}, pos={sum(df.target==1)}, neg={sum(df.target==0)}")

    train_df, test_df = stratified_split(df, 0.1, cfg.random_seed)

    # training
    print("===== K‑Fold Training =====")
    cross_validate(train_df, cfg)

    print("\n===== Hold‑out Evaluation =====")
    evaluate_test(test_df, cfg)

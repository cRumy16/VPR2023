import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help="The number of places to use per iteration (one place is N images)")
    parser.add_argument("--img_per_place", type=int, default=4,
                        help="The effective batch size is (batch_size * img_per_place)")
    parser.add_argument("--min_img_per_place", type=int, default=4,
                        help="places with less than min_img_per_place are removed")
    parser.add_argument("--max_epochs", type=int, default=20,
                        help="stop when training reaches max_epochs")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of processes to use for data loading / preprocessing")
    parser.add_argument("--load_checkpoint", default=False,
                        help="whether to load pytorch lightning checkpoints")
    parser.add_argument("--only_test", default=False,
                        help="avoid the train phase and test on --test_path")
    parser.add_argument("--self_supervised", default=False,
                        help="self_supervised enable")
    
    # Architecture parameters
    parser.add_argument("--descriptors_dim", type=int, default=512,
                        help="dimensionality of the output descriptors")
    parser.add_argument("--enable_gpm", default=False,
                        help="whether to adopt Global Proxy Mining module")
    parser.add_argument("--pooling_layer", type = str, default=None,
                        help="change the last pooling layer")
    parser.add_argument("--optimizer", type = str, default="sgd",
                        help="change the optimizer")
    parser.add_argument("--lr_scheduler", type = str, default="",
                        help="change the learning rate scheduler")
    
    # Visualizations parameters
    parser.add_argument("--num_preds_to_save", type=int, default=0,
                        help="At the end of training, save N preds for each query. "
                        "Try with a small number like 3")
    parser.add_argument("--save_only_wrong_preds", action="store_true",
                        help="When saving preds (if num_preds_to_save != 0) save only "
                        "preds for difficult queries, i.e. with uncorrect first prediction")

    # Paths parameters
    parser.add_argument("--train_path", type=str, default="data/gsv_xs/train",
                        help="path to train set")
    parser.add_argument("--val_path", type=str, default="data/sf_xs/val",
                        help="path to val set (must contain database and queries)")
    parser.add_argument("--test_path", type=str, default="data/sf_xs/test",
                        help="path to test set (must contain database and queries)")
    parser.add_argument("--checkpoint_path", type=str, default="./LOGS/lightning_logs/version_0/checkpoints/*.ckpt",
                        help="path for loading pytorch lightning checkpoints")

    args = parser.parse_args()
    return args


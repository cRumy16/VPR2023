import torch
import numpy as np
import torchvision.models
import pytorch_lightning as pl
import torchvision.transforms as tfm
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

import utils
import parser
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset


class LightningModel(pl.LightningModule):
    def __init__(self, val_dataset, test_dataset, avgpool, avgpool_param = {}, 
                proxy_bank = None, descriptors_dim = 512, num_preds_to_save = 0, save_only_wrong_preds = True, self_supervised=False, optimizer_choice = "sgd", lr_scheduler = None):
        super().__init__()
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_preds_to_save = num_preds_to_save
        self.save_only_wrong_preds = save_only_wrong_preds
        self.self_supervised = self_supervised

        self.optimizer_choice = optimizer_choice
        self.lr_scheduler = lr_scheduler
        self.milestones = [5, 10, 15]
        # Use a pretrained model
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        
        if avgpool == "GeM":
            self.model.avgpool = utils.GeM()
        elif avgpool == "CosPlace":
            avgpool_param = {'in_dim': 512, 'out_dim': 512}
            self.model.avgpool = utils.CosPlace(avgpool_param['in_dim'], avgpool_param['out_dim'])
        elif avgpool == "mixvpr":
            print("Add: mixvpr")
            self.mixvpr_out_channels = 256
            self.mixvpr_out_rows = 4
            # MixVPR works with an input of dimension [n_batch, 512, 7,7] == [n_batch, in_channels, in_h, in_w]
            self.model.avgpool = utils.MixVPR( in_channels = self.model.fc.in_features, in_h = 7, in_w = 7, out_channels = self.mixvpr_out_channels , out_rows =  self.mixvpr_out_rows )
        
        # Initialize output dim as the standard one of CNN
        self.aggregator_out_dim = self.model.fc.in_features

        if avgpool == "mixvpr":
            self.aggregator_out_dim  = self.mixvpr_out_channels * self.mixvpr_out_rows
            self.model.fc = torch.nn.Linear(self.aggregator_out_dim, descriptors_dim)
        else:
             self.model.fc = torch.nn.Linear(self.model.fc.in_features, descriptors_dim)
        
        # Instantiate the Proxy Head and Proxy Bank
        self.pbank = proxy_bank #non serve nell'if, al massimo = None
        if args.enable_gpm:
            self.phead = utils.ProxyHead(args.descriptors_dim)
            self.loss_head = losses.MultiSimilarityLoss(alpha=1, beta=50, base=0.0)
            # self.loss_head = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        if self.self_supervised:
            self.loss_aug = losses.VICRegLoss(invariance_lambda=25, variance_mu=25, covariance_v=1, eps=1e-4)
        # Set miner
        # self.miner_fn = miners.MultiSimilarityMiner(epsilon=0.1)
        # Set loss_function
        # self.loss_fn = losses.MultiSimilarityLoss(alpha=1, beta=50, base=0.0)
        self.loss_fn = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)

    def forward(self, images, is_transformed):
        descriptors = self.model(images)
        if args.enable_gpm:
            compressed_descriptors = self.phead(descriptors)
        else:
            compressed_descriptors = None
        if is_transformed:
            # descriptors = self.model(images)
            return descriptors
        return descriptors, compressed_descriptors

    def configure_optimizers(self):
        if self.optimizer_choice == "sgd":
            optimizers = torch.optim.SGD(self.parameters(), lr=0.001, weight_decay=0.001, momentum=0.9)
        if self.optimizer_choice == "adam":
            print("Add: ", self.optimizer_choice)
            optimizers = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        #TODO: change
        if self.lr_scheduler == "reducelronplateau" :
            print("Add: ", self.lr_scheduler)
            schedulers = [ torch.optim.lr_scheduler.MultiStepLR(optimizers, milestones=self.milestones, gamma=0.3) ]
        else :
            schedulers = []
        return [optimizers] , schedulers

    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # Include a miner for loss'pair selection
        # miner_output = self.miner_fn(descriptors , labels)
        # Compute the loss using the loss function and the miner output instead of all possible batch pairs
        # loss = self.loss_fn(descriptors, labels, miner_output)
        loss = self.loss_fn(descriptors, labels)
        return loss
    
    def vicreg_loss(self, descriptors, labels, ref_desc):
        vicreg_loss = self.loss_aug(descriptors, ref_emb = ref_desc)
        return vicreg_loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        if self.self_supervised:
            images, images_aug, labels = batch
        else:
            images, labels = batch
        num_places, num_images_per_place, C, H, W = images.shape
        images = images.view(num_places * num_images_per_place, C, H, W)
        images_aug = images_aug.view(num_places * num_images_per_place, C, H, W)
        labels = labels.view(num_places * num_images_per_place)
        # Feed forward the batch to the model
        descriptors, compressed_descriptors = self(images, False)  # Here we are calling the method forward that we defined above
        descriptors_aug, compressed_descriptors = self(images_aug,False)
        loss = self.loss_function(descriptors, labels)  # Call the loss_function we defined above

        if self.self_supervised:
            loss = self.vicreg_loss(descriptors=descriptors, labels=labels, ref_desc = descriptors_aug)

        if args.enable_gpm:
            # descriptors = descriptors.cpu() #tensore privo di gradient
            # compressed_descriptors = compressed_descriptors.cpu().detach()
            self.pbank.update_bank(compressed_descriptors, labels)
            loss_head = self.loss_head(compressed_descriptors, labels)
            loss = loss + loss_head
        
        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}
    # For validation and test, we iterate step by step over the validation set
    def inference_step(self, batch):
        images, _ = batch
        descriptors, _ = self(images, False)
        return descriptors.cpu().numpy().astype(np.float32)

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def validation_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.val_dataset)

    def test_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.test_dataset, self.num_preds_to_save)

    def inference_epoch_end(self, all_descriptors, inference_dataset, num_preds_to_save=0):
        if self.pbank is not None:
            self.pbank.update_index()
        """all_descriptors contains database then queries descriptors"""
        all_descriptors = np.concatenate(all_descriptors)
        queries_descriptors = all_descriptors[inference_dataset.database_num : ]
        database_descriptors = all_descriptors[ : inference_dataset.database_num]

        recalls, recalls_str = utils.compute_recalls(
            inference_dataset, queries_descriptors, database_descriptors,
            trainer.logger.log_dir, num_preds_to_save, self.save_only_wrong_preds
        )
        print(recalls_str)
        self.log('R@1', recalls[0], prog_bar=False, logger=True)
        self.log('R@5', recalls[1], prog_bar=False, logger=True)

def get_datasets_and_dataloaders(self, args, bank=None):
    if self.self_supervised : 
        train_transform = tfm.Compose([
             tfm.RandomApply(transforms=[
                    tfm.RandomHorizontalFlip(p = 0.7),
                    tfm.RandomCrop(size=224),
            ], p=0.3),
            tfm.ToTensor(),
            tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = tfm.Compose([
            tfm.RandAugment(num_ops=3),
            tfm.ToTensor(),
            tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    train_dataset = TrainDataset(
        dataset_folder=args.train_path,
        img_per_place=args.img_per_place,
        min_img_per_place=args.min_img_per_place,
        transform=train_transform,
        self_supervised=args.self_supervised
    )
    val_dataset = TestDataset(dataset_folder=args.val_path)
    test_dataset = TestDataset(dataset_folder=args.test_path)

    # Define dataloaders, train one has with proxy and without proxy case
    if bank is not None:
        # Proxy Sampler with ProxyBank
        my_proxy_sampler = utils.ProxyBankBatchSampler(train_dataset, args.batch_size , bank)
        train_loader = DataLoader(dataset=train_dataset, batch_sampler = my_proxy_sampler, num_workers=args.num_workers)
    else:
        if self.self_supervised:
            train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        else :
            train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
   
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader

if __name__ == '__main__':
    args = parser.parse_arguments()

    if args.enable_gpm:
        print("Add: gpm")
        proxy_bank = utils.ProxyBank(proxy_dim=512)
    else:
        proxy_bank = None

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_datasets_and_dataloaders(args, proxy_bank)
    kwargs = {"val_dataset": val_dataset, "test_dataset": test_dataset, "avgpool": args.pooling_layer, "self_supervised": args.self_supervised, "optimizer_choice": args.optimizer, "lr_scheduler": args.lr_scheduler}
    
    if args.enable_gpm:
        kwargs.update({"proxy_bank": proxy_bank})
    
    if args.load_checkpoint:
        model = LightningModel.load_from_checkpoint(args.checkpoint_path, **kwargs)
    else:
        model = LightningModel(**kwargs)

    # Model params saving using Pytorch Lightning. Save the best 3 models according to Recall@1
    checkpoint_cb = ModelCheckpoint(
        monitor='R@1',
        filename='_epoch({epoch:02d})_step({step:04d})_R@1[{val/R@1:.4f}]_R@5[{val/R@5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode='max'
    )

    # Instantiate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        default_root_dir='./LOGS',  # Tensorflow can be used to viz
        num_sanity_val_steps=0,  # runs a validation step before stating training
        precision=16,  # we use half precision to reduce  memory usage
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,  # run validation every epoch
        callbacks=[checkpoint_cb],  # we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=20,
    )
    
    # Train or test only with a pretrained model
    if not args.only_test:
        trainer.validate(model=model, dataloaders=val_loader)
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=model, dataloaders=test_loader)

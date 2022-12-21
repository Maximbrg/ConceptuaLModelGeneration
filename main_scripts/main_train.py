import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import lr_scheduler
import time

from handle_data.handle_data_modelSet import CustomModelSetDataset
from models.network_SAGEConv import Net

class BuildModel:
    def __init__(self, net, loss_fn_type: str = "nll_loss", optimizer_type: str = "adam",
                 G_optimizer_wd: int = 0, optimizer_lr: float = 0.0002, optimizer_betas: list = [0.9, 0.999],
                 save_dir: str = "../results",
                 train_loader: object = None, epochs: int = 30, checkpoint_save: int = 200):
        """

        :param net:
        :param loss_fn_type:
        :param optimizer_type:
        :param optimizer_lr:
        :param optimizer_betas:
        :param save_dir:
        :param train_loader:
        :param epochs:
        :param checkpoint_save:
        """

        self.y = None
        self.G_loss = None
        self.E = None
        self.edge_index = None
        self.x = None
        self.G_optimizer = None
        self.G_loss_fn = None
        self.net = net
        self.loss_fn_type = loss_fn_type
        self.optimizer_type = optimizer_type
        self.optimizer_lr = optimizer_lr
        self.G_optimizer_betas = optimizer_betas
        self.schedulers = []
        self.train_loader = save_dir
        self.G_optimizer_wd = G_optimizer_wd
        self.checkpoint_save = checkpoint_save
        self.epochs = epochs
        self.train_loader = train_loader

    def init_train(self):
        self.netG.train()
        self.define_loss()
        self.define_optimizer()
        self.define_scheduler()

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        if self.G_lossfn_type == 'nll_loss':
            self.G_loss_fn = nn.functional.nll_loss.L1Loss()
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(self.G_lossfn_type))

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))

        self.G_optimizer = Adam(G_optim_params, lr=self.G_optimizer_lr,
                                betas=self.G_optimizer_betas,
                                weight_decay=self.G_optimizer_wd)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        milestones=[1, 2, 3],
                                                        gamma=0.1))

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)

    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data):
        print("datat[L] shape", data["x"].shape)
        self.x = data['L']
        self.edge_index = data['edge_index']
        self.y = data['Y']

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.x, self.edge_index)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        self.G_loss = self.G_lossfn(self.E, self.y)
        self.G_loss.backward()
        self.G_optimizer.step()

    def fit(self):
        self.init_train()
        current_step = 0

        for epoch in range(self.epochs):
            for i, train_data in enumerate(self.train_loader):
                st = time.time()

                current_step += 1

                # -------------------------------
                # 1) update learning rate
                # -------------------------------
                self.update_learning_rate(current_step)
                lr = self.get_lr()  # get learning rate

                # -------------------------------
                # 2) feed patch pairs
                # -------------------------------
                self.feed_data(train_data)

                # -------------------------------
                # 3) optimize parameters
                # -------------------------------
                self.optimize_parameters(current_step)

                # -------------------------------
                # 6) Save model
                # -------------------------------
                # if current_step == 1 or current_step % checkpoint_save == 0:
                #     self.save(current_step)
                #     print("Model Loss {} after step {}".format(self.G_loss, current_step))
                #     print("Model Saved")
                #     print("Time per step:", time.time() - st)
                #     wandb.log({"loss": self.G_loss, "lr": lr})


def run(file_path: str = 'C:\\Users\\max_b\\PycharmProjects\\ConceptuaLModelGeneration\\dataset_1'):

    dataset = CustomModelSetDataset(dataset_dir=file_path, nlp_type='en_core_web_lg')
    model = Net(num_features=300, num_classes=None)
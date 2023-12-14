import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn
import torch.nn.functional as F
selected_d = {"outs": [], "trg": [],"probs": []}

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, criterion_MSE_m1, criterion_MSE_m2, criterion_Feature_Level, criterion_Context_Level, criterion_MSE_P1, criterion_MSE_P2, metrics_ftns, optimizer, config, data_loader, fold_id,
                 valid_data_loader=None, class_weights=None):
        super().__init__(model, criterion, metrics_ftns, optimizer, config, fold_id)
        #self.model = model
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.criterion_MSE_m1 = criterion_MSE_m1
        self.criterion_MSE_m2 = criterion_MSE_m2
        self.criterion_Feature_Level = criterion_Feature_Level
        self.criterion_Context_Level = criterion_Context_Level
        self.criterion_MSE_P1 = criterion_MSE_P1
        self.criterion_MSE_P2 = criterion_MSE_P2
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = optimizer
        self.log_step = int(data_loader.batch_size) * 1  # reduce this if you want more logs
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.train_metrics0 = MetricTracker('loss_ce', *[m.__name__ for m in self.metric_ftns])
        self.train_metrics1 = MetricTracker('loss_MSEm1', *[m.__name__ for m in self.metric_ftns])
        self.train_metrics2 = MetricTracker('loss_MSEm2', *[m.__name__ for m in self.metric_ftns])
        self.train_metrics3 = MetricTracker('loss_FL', *[m.__name__ for m in self.metric_ftns])
        self.train_metrics4 = MetricTracker('loss_CL', *[m.__name__ for m in self.metric_ftns])
        self.train_metrics5 = MetricTracker('loss_P1', *[m.__name__ for m in self.metric_ftns])
        self.train_metrics6 = MetricTracker('loss_P2', *[m.__name__ for m in self.metric_ftns])

        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.device = torch.device('cuda:0')
        self.fold_id = fold_id
        self.selected = 0
        self.class_weights = class_weights
    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
               total_epochs: Integer, the total number of epoch
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.train_metrics1.reset()
        self.train_metrics2.reset()
        self.train_metrics3.reset()
        overall_outs = []
        overall_trgs = []
        overall_probs = []

        for batch_idx, (data_m1, data_m2, data_label) in enumerate(self.data_loader):
            data_m1, data_m2, data_label = data_m1.float().to(self.device), data_m2.float().to(self.device), data_label.float().to(self.device)
            self.optimizer.zero_grad()

            out1, out2 = self.model.MultiEncoder(data_m1, data_m2)
            mask_m1 = torch.ones((out1.shape[0], out1.shape[1], out1.shape[2])).to(self.device)
            mask_m2 = torch.zeros((out2.shape[0], out2.shape[1], out2.shape[2])).to(self.device)
            out_MI1 = self.model.projection_MI1(out1)
            out_MI2 = self.model.projection_MI2(out2)
            out_MKLD1 = self.model.projection_MKLD1(out1)
            out_MKLD2 = self.model.projection_MKLD2(out2)
            #out3, out4 = self.model.Imputation(out_MKLD1, out_MKLD2, mask_m1, mask_m2)
            out_m1_to_m2 = self.model.imputation1(out1)
            out_m2_to_m1 = self.model.imputation2(out2)
            # out_MKLD11 = out1 * mask_m1 + out_m2_to_m1 * mask_m2
            # out_MKLD22 = out2 * mask_m2 + out_m1_to_m2 * mask_m1
            out_MKLD11 = out1 * mask_m1 + out_m2_to_m1 * mask_m2
            out_MKLD22 = out2 * mask_m1 + out_m1_to_m2 * mask_m2
            out5 = self.model.cross_tce1(out_MKLD11, out_MKLD22, out_MKLD22)
            out6 = self.model.cross_tce2(out_MKLD22, out_MKLD11, out_MKLD11)
            out_cat = torch.cat([out5, out6], dim=2)
            out_cat_T = self.model.tce(out_cat)
            out_cla = self.model.classfier(out_cat_T)
            out7, out8 = self.model.MultiDecoder(out_MKLD11, out_MKLD22)
            #print("out7",out7)


            out_MI1 = out_MI1.to(self.device)
            out_MI2 = out_MI2.to(self.device)
            out_MKLD1 = out_MKLD1.to(self.device)
            out_MKLD2 = out_MKLD2.to(self.device)
            out_cla = out_cla.to(self.device)
            out7 = out7.to(self.device)
            out8 = out8.to(self.device)
            out_m1_to_m2 = out_m1_to_m2.to(self.device)
            out_m2_to_m1 = out_m2_to_m1.to(self.device)



            out_MKLD1 = out_MKLD1
            out_MKLD2 = out_MKLD2
            out_cla = out_cla.flatten(end_dim=1)
            out7 = out7.flatten(end_dim=1)
            out8 = out8.flatten(end_dim=1)


            loss_ce = self.criterion(out_cla, data_label.flatten()).to(self.device)
            #loss_mse_m1 = self.criterion_MSE_m1(out7, data_m1.flatten(end_dim=1))
            #loss_mse_m2 = self.criterion_MSE_m2(out8, data_m2.flatten(end_dim=1))
            loss_mse_m1 = F.mse_loss(out7, data_m1.flatten(end_dim=1))
            loss_mse_m2 = F.mse_loss(out8, data_m2.flatten(end_dim=1))
            loss_FL = self.criterion_Feature_Level(out_MI1, out_MI2)
            loss_CL = self.criterion_Context_Level(out_MKLD1, out_MKLD2)
            # loss_mse_p1 = self.criterion_MSE_P1(out_m1_to_m2, out_MKLD2)
            # loss_mse_p2 = self.criterion_MSE_P1(out_m2_to_m1, out_MKLD1)
            #loss_mse_p1 = F.mse_loss(out_m1_to_m2, out_MKLD2)
            #loss_mse_p2 = F.mse_loss(out_m2_to_m1, out_MKLD1)
            #loss_pp = loss_mse_p1 + loss_mse_p2
            #print(loss_pp)
            loss = loss_ce + 1 * loss_mse_m1 + 1 * loss_mse_m2 + 0.1 * loss_FL + 0.1 * loss_CL
            # if epoch >= 20:
            #     loss += 0.1 * loss_pp

            # print(loss_ce)
            # print(loss_mse_m1)
            # print(loss_mse_m2)
            # print(loss_FL)
            # print(loss_CL)

            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            self.train_metrics0.update('loss_ce', loss_ce.item())
            self.train_metrics1.update('loss_MSEm1', loss_mse_m1.item())
            self.train_metrics2.update('loss_MSEm2', loss_mse_m2.item())
            self.train_metrics3.update('loss_FL', loss_FL.item())
            self.train_metrics4.update('loss_CL', loss_CL.item())
            #self.train_metrics5.update('loss_P1', loss_mse_p1.item())
            #self.train_metrics6.update('loss_P2', loss_mse_p2.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(out_cla, data_label.flatten()))

            if batch_idx % self.log_step == 0:
                #print(self.criterion.epsilon.grad)
                # 查看 epsilon 的梯度
                #epsilon_grad = self.criterion.epsilon.grad.item()
                #print(epsilon_grad)

                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    loss_ce.item(),
                    loss_mse_m1.item(),
                    loss_mse_m2.item(),
                    loss_FL.item(),
                    loss_CL.item()
                    #loss_mse_p1.item(),
                    #loss_mse_p2.item()

                ))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        log0 = self.train_metrics0.result()
        log1 = self.train_metrics1.result()
        log2 = self.train_metrics2.result()
        log3 = self.train_metrics3.result()
        log4 = self.train_metrics4.result()

        if self.do_validation:
            val_log, outs, trgs, probs = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if val_log["accuracy"] > self.selected:
                self.selected = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs
                selected_d["probs"] = probs
            if epoch == total_epochs:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])
                overall_probs.extend(selected_d["probs"])
            # THIS part is to reduce the learning rate after 10 epochs to 1e-4
            if epoch == 10:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.0001


        return log, overall_outs, overall_trgs, overall_probs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            probs = np.array([])
            for batch_idx, (data_m1, data_m2, data_label) in enumerate(self.valid_data_loader):
                data_m1, data_m2, data_label = data_m1.float().to(self.device), data_m2.float().to(self.device), data_label.float().to(self.device)


                out1, out2 = self.model.MultiEncoder(data_m1, data_m2)
                # out3, out4 = self.model.Imputation(out_MKLD1, out_MKLD2, mask_m1, mask_m2)
                #out_m2_to_m1 = self.model.imputation2(out2)

                out5 = self.model.cross_tce1(out1, out2, out2)
                out6 = self.model.cross_tce2(out2, out1, out1)
                out_cat = torch.cat([out5, out6], dim=2)
                out_cat_T = self.model.tce(out_cat)
                out_cla = self.model.classfier(out_cat_T)


                out_cla = out_cla.to(self.device)
                out_cla = out_cla.flatten(end_dim=1)

                #output = output.to(self.device)
                #output = output + self.adjustment
                #loss = self.criterion(output, target, self.class_weights, self.device)
                #self.criterion.set_epsilon(A)
                loss = self.criterion(out_cla, data_label.flatten()).to(self.device)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(out_cla, data_label.flatten()))

                preds_ = out_cla.data.max(1, keepdim=True)[1].cpu()
                probs = np.append(probs, out_cla.data.cpu().numpy())
                outs = np.append(outs, preds_.cpu().numpy())
                trgs = np.append(trgs, data_label.data.cpu().numpy())


        return self.valid_metrics.result(), outs, trgs,probs

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
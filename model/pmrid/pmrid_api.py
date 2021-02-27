import os
import cv2
import skimage.metrics
import numpy as np
from typing import Tuple
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.pmrid.utils import RawUtils
from model.pmrid.pmrid import PMRID

class KSigma:
    def __init__(self, k_coeff: Tuple[float, float], b_coeff: Tuple[float, float, float], anchor: float, v: float = 959.0):
        self.K = np.poly1d(k_coeff)
        self.Sigma = np.poly1d(b_coeff)
        self.anchor = anchor
        self.v = v

    def __call__(self, img_01, iso: float, inverse=False):
        k, sigma = self.K(iso), self.Sigma(iso)
        k_a, sigma_a = self.K(self.anchor), self.Sigma(self.anchor)

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        img = img_01 * self.v

        if not inverse:
            img = img * cvt_k + cvt_b
        else:
            img = (img - cvt_b) / cvt_k

        return img / self.v



class DataProcess():
    def __init__(self):
        self.k_sigma = KSigma(
            k_coeff=[0.0005995267, 0.00868861],
            b_coeff=[7.11772e-7, 6.514934e-4, 0.11492713],
            anchor=1600,
        )
    
    def pre_process(self, bayer: np.ndarray, iso: float):
        # normalize
        bayer = bayer / 255.0

        # bayer to rggb
        rggb = RawUtils.bayer2rggb(bayer)
        rggb = rggb.clip(0, 1)

        # padding
        H, W = rggb.shape[:2]
        ph, pw = (32-(H % 32))//2, (32-(W % 32))//2
        self.ph, self.pw = ph, pw
        rggb = np.pad(rggb, [(ph, ph), (pw, pw), (0, 0)], 'constant')

        # transpose
        rggb = rggb.transpose(2, 0, 1)
        
        # ksigma
        rggb = self.k_sigma(rggb, iso)
        
        # inverse normalize
        rggb = rggb * 255.0
        return rggb

    def post_process(self, rggb: np.ndarray, iso: float):
        # normalize
        rggb = rggb / 255.0

        # ksigma
        rggb = self.k_sigma(rggb, iso, inverse = True)

        # transpose
        rggb = rggb.transpose(1, 2, 0)

        # inverse padding
        ph, pw = self.ph, self.pw
        rggb = rggb[ph:-ph, pw:-pw]

        # rggb to bayer
        bayer = RawUtils.rggb2bayer(rggb)
        bayer = bayer.clip(0, 1)

        #  inverse normalize
        bayer = bayer * 255.0
        return bayer 



class PMRIDDataset(Dataset):
    def __init__(self, filepath, data_process, train = False):
        self.input_path = []
        self.gt_path = []
        for line in open(filepath):
            self.input_path.append(line.split(" ")[0])
            self.gt_path.append(line.split(" ")[1])
        self.len = len(self.input_path)
        self.data_process = data_process

    def __getitem__(self, index):
        input_iso = 4300
        gt_iso = 4300

        input_bayer = cv2.imread(self.input_path[index], 0).astype(np.float32)
        input_rggb = self.data_process.pre_process(input_bayer, input_iso)
        input_data = torch.from_numpy(input_rggb)  

        gt_bayer = cv2.imread(self.gt_path[index], 0).astype(np.float32)       
        gt_rggb = self.data_process.pre_process(gt_bayer, gt_iso)
        gt_data = torch.from_numpy(gt_rggb)

        label = self.input_path[index].split('/')[-1]
        return input_data, gt_data, input_iso, gt_iso, label

    def __len__(self):
        return self.len


class PMRID_API():
    def __init__(self, epoch, batch_size, learning_rate, device, logs_path, params_path, train_list_path, value_list_path, is_load_pretrained, pretrained_path):
        # parameters
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.logs_path = logs_path
        self.params_path = params_path        
        self.train_list_path = train_list_path
        self.value_list_path = value_list_path
        self.is_load_pretrained = is_load_pretrained
        self.pretrained_path = pretrained_path

        # data process
        self.data_process = DataProcess()

        # data loader
        train_dataset = PMRIDDataset(self.train_list_path, self.data_process)        
        value_dataset = PMRIDDataset(self.value_list_path, self.data_process)
        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        self.value_loader = DataLoader(value_dataset, shuffle=False, batch_size=1)

        # tensorboard
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        self.writer = SummaryWriter(self.logs_path)

        # networks
        if not os.path.exists(self.params_path):
            os.makedirs(self.params_path)
        self.pmrid = PMRID()
        # self.criterion = torch.nn.L1Loss()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.pmrid.parameters(), lr=self.learning_rate)        

    def load_weight(self, path):
        states = torch.load(path)
        self.pmrid.load_state_dict(states)
        self.pmrid.to(self.device)
        print('[load] load finished')

    def init_weight(self):
        pass

    def train(self, epoch):
        losses = []
        for batch_idx, data in enumerate(self.train_loader, 0):
            inputs, gts, input_iso, gt_iso, label = data
            inputs, gts = inputs.to(self.device), gts.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.pmrid(inputs)
            loss = self.criterion(outputs, gts)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())            
            print('[train] epoch: %d, batch: %d, loss: %f' % (epoch + 1, batch_idx + 1, loss.item()))
        
        mean_loss = np.mean(losses)
        print('[value] epoch: %d, mean loss: %f' % (epoch + 1, mean_loss))
        self.writer.add_scalar('loss', mean_loss, epoch + 1)

    def value(self, epoch):
        psnrs = []
        ssims = []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.value_loader, 0):
                inputs, gts, input_iso, gt_iso, label = data
                inputs = inputs.to(self.device)
                gts = inputs.to(self.device)

                # run pmrid
                outputs = self.pmrid(inputs)

                input_rggb = inputs.squeeze().cpu().numpy()
                input_bayer = self.data_process.post_process(input_rggb, input_iso[0]) / 255.0

                gt_rggb = gts.squeeze().cpu().numpy()
                gt_bayer = self.data_process.post_process(gt_rggb, gt_iso[0]) / 255.0

                output_rggb = outputs.squeeze().cpu().numpy()
                output_bayer = self.data_process.post_process(output_rggb, input_iso[0]) / 255.0       

                psnr = skimage.metrics.peak_signal_noise_ratio(gt_bayer, output_bayer)
                ssim = skimage.metrics.structural_similarity(gt_bayer, output_bayer)
                psnrs.append(float(psnr))
                ssims.append(float(ssim))
                print('[value] epoch: %d, batch: %d, pnsr: %f, ssim: %f' % (epoch + 1, batch_idx + 1, psnr, ssim))

        mean_psnr = np.mean(psnrs)
        mean_ssim = np.mean(ssims)
        print('[value] epoch: %d, mean pnsr: %f, mean ssim: %f' % (epoch + 1, mean_psnr, mean_ssim))
        self.writer.add_scalar('psnr', mean_psnr, epoch + 1)
        self.writer.add_scalar('ssim', mean_ssim, epoch + 1)

    def train_and_value(self):
        if self.is_load_pretrained:
            self.load_weight(self.pretrained_path)
        else:
            self.init_weight()
        for epoch in range(self.epoch):
            if self.is_load_pretrained:
                pass
            else:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate * (0.5 ** (epoch // 20))
            self.train(epoch)
            self.value(epoch)
            torch.save(self.pmrid.state_dict(), self.params_path+'/'+str(epoch+1)+'.ckp')

    def test(self, params_path, output_path):
        self.load_weight(params_path)
        with torch.no_grad():
            for batch_idx, data in enumerate(self.value_loader, 0):
                inputs, gts, input_iso, gt_iso, label = data
                inputs = inputs.to(self.device)
                gts = inputs.to(self.device)

                # run pmrid
                outputs = self.pmrid(inputs)

                output_rggb = outputs.squeeze().cpu().numpy()
                output_bayer = self.data_process.post_process(output_rggb, input_iso[0])

                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                print('[test] ' +output_path + label[0])
                cv2.imwrite(output_path + label[0], output_bayer.astype(np.uint8))


if __name__=='__main__':
    pmrid_api = PMRID_API(
        200, 
        10,
        0.01,
        'cuda:0',
        './logs/pmrid_l2/',
        './params/pmrid_l2/',
        './data/right_value_list.txt',
        './data/right_value_list.txt',
        True,
        './model/pmrid/pmrid_pretrained.ckp'
        )
    pmrid_api.train_and_value()























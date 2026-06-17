import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch.utils import data, tensorboard
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

import copy
from pathlib import Path
from einops import rearrange

from cardiac_diffusion.dataset import CMRxProcessed
from cardiac_diffusion.utils import normalize


def cycle(dl):
    # creates an infinite loop over a given DataLoader
    # as the number of training steps is specified rather than the number of epochs
    while True:
        for inputs in dl:
            yield inputs


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            train_path,
            *,
            log_dir,
            model_dir,
            val_path=None, 
            dataset='CMRxProcessed',
            unet_type='SpatioTemporalUnet',
            multicoil=True,
            filter='',
            model_path=None,
            resume_training=False,
            train_steps=300000,
            batch_size=1,
            lr=2e-5,
            gradient_accumulate_every=2,
            loss_type='l2',
            optimizer='adam',
            scheduler=False,
            step_start_ema=2000,
            update_ema_every=20,
            ema_decay=0.995,
            sample_and_save_every=1000,
            eta=0.95,
            train_only_first=None,
            randomize_cardiac_cycle=True,
            **kwargs
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every
        self.reset_parameters()
        self.step = 0
        self.milestone = 0

        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_steps = train_steps
        self.train_only_first = train_only_first

        self.sample_and_save_every = sample_and_save_every
        self.eta = eta

        self.loss_type = loss_type
        if loss_type == 'l1':
            loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            loss_fn = nn.MSELoss()
        elif loss_type == 'l2-magnitude':
            to_mag = lambda x: abs(torch.complex(x[:, 0], x[:, 1]))
            loss_fn = lambda x, y: nn.MSELoss()(to_mag(x), to_mag(y))
        else:
            raise NotImplementedError("Loss type not implemented. Default: l2")
        
        self.randomize_cardiac_cycle = randomize_cardiac_cycle 
        
        if unet_type == 'MiddleFrameUnet' or unet_type == 'MiddleFrameAsymmetricUnet':
            self.predictMiddleFrame = True
            takeMiddleFrame = lambda x: x[:, :, x.shape[2] // 2, :, :]
            self.loss_fn = lambda x, y: loss_fn(takeMiddleFrame(x), y)
            self.slab_size = self.model.module.restoration_fn.input_frames
        else:
            self.predictMiddleFrame = False
            self.loss_fn = loss_fn
        
        if dataset == 'CMRxProcessed':
            self.dataset = CMRxProcessed
        else:
            raise NotImplementedError("Dataset type not implemented. Default: CMRxProcessed")

        self.ds = self.dataset(train_path, filter=filter, multicoil=multicoil, image_only=True)
        self.train_sampler = data.DistributedSampler(self.ds)
        self.dl = cycle(
            data.DataLoader(self.ds,
                            batch_size=batch_size,
                            sampler=self.train_sampler,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=8,
                            drop_last=False,
                            ) 
                        )

        if val_path is not None:
            self.vds = self.dataset(val_path, filter=filter, multicoil=multicoil, image_only=True)
            self.val_sampler = data.DistributedSampler(self.vds)
            self.vdl = cycle(
                data.DataLoader(self.vds,
                                batch_size=1,
                                sampler=self.val_sampler,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=8,
                                drop_last=True,
                                )
                            )
        else:
            self.vdl = self.dl
            print('\nValidation data not provided. Using training data for purpose of plotting progress.\n')
        
        if optimizer == 'adam':
            self.opt = Adam(diffusion_model.parameters(), lr=lr)
        else:
            raise NotImplementedError("Optimizer type not implemented. Default: adam")
        
        self.resume_training = resume_training
        if model_path is not None:
            self.load(model_path)

        if scheduler:
            last_epoch = self.step if resume_training else -1
            self.scheduler = OneCycleLR(
                optimizer=self.opt, 
                max_lr=lr,
                total_steps=self.train_steps+1,
                pct_start=0.1,
                anneal_strategy='cos',
                cycle_momentum=False,
                base_momentum=0., 
                max_momentum=0.,
                div_factor = 20,
                final_div_factor=100,
                last_epoch=last_epoch
            )
        else:
            self.scheduler = None
        
        self.model_dir = model_dir
        self.writer = tensorboard.SummaryWriter(log_dir=log_dir)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, path, identifier=''):
        model_data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'opt': self.opt.state_dict()
        }
        torch.save(model_data, str(path / f'{identifier}model.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        rank = dist.get_rank()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model_data = torch.load(load_path, map_location=map_location) 

        if self.resume_training:
            self.step = model_data['step']
            self.opt.load_state_dict(model_data['opt'])

        self.model.load_state_dict(model_data['model'])
        self.ema_model.load_state_dict(model_data['ema'])
        self.milestone = int(Path(load_path).stem.split('-')[0])

    def choose_slab(self, x):
        q = torch.randint(0, x.shape[2], (1,))
        start, stop = q - self.slab_size // 2, q + self.slab_size // 2 + self.slab_size % 2

        # only works for slab_size <= 2 * x.shape[2]
        # we also want it to start at different positions in the cycle
        if self.slab_size > x.shape[2]:
            x = F.pad(x, (0, 0, 0, 0, self.slab_size-x.shape[2], 0), mode='circular')
            start, stop = 0, self.slab_size
        elif start < 0:
            x = F.pad(x, (0, 0, 0, 0, abs(start), 0), mode='circular')
            start, stop = 0, self.slab_size    
        elif stop > x.shape[2]:
            x = F.pad(x, (0, 0, 0, 0, 0, stop-x.shape[2]), mode='circular')
            start, stop = x.shape[2] - self.slab_size, x.shape[2]
    
        return x[:, :, start:stop]

    def train(self):
        acc_loss = 0

        # enable mixed-precision training: torch.amp
        scaler = torch.amp.GradScaler("cuda")

        # iterate for a fixed number of optimization steps
        while self.step <= self.train_steps:
            self.opt.zero_grad()
            u_loss = 0
            for _ in range(self.gradient_accumulate_every):
                inputs = next(self.dl).cuda()

                if self.train_only_first is not None:
                    assert self.train_only_first > 0 and self.train_only_first < 1, "train_only_first must be in the range (0, 1), or None"
                    t = torch.randint(0, int(self.model.module.timesteps * self.train_only_first), (inputs.shape[0],)).cuda()
                else:
                    t = None

                if self.randomize_cardiac_cycle:
                    self.slab_size = 12
                    inputs = self.choose_slab(inputs)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    if self.model.module.noise_based:
                        noise = torch.randn_like(inputs)
                        outputs = self.model(inputs, t=t, noise=noise)
                        loss = torch.mean(self.loss_fn(noise, outputs))
                    else:
                        outputs = self.model(inputs, t=t)
                        loss = torch.mean(self.loss_fn(inputs, outputs))

                u_loss += loss.item()
                scaler.scale(loss / self.gradient_accumulate_every).backward()

            if dist.get_rank() == 0:
                self.writer.add_scalar('Loss/training', u_loss / self.gradient_accumulate_every, self.step)
            print(f'{self.step}: {u_loss / self.gradient_accumulate_every}')
            acc_loss = acc_loss + (u_loss / self.gradient_accumulate_every)

            scaler.step(self.opt)
            scaler.update()
            
            if self.scheduler is not None:
                self.scheduler.step()

            if self.step % 50 == 0:
                with torch.no_grad():
                    rep = 16
                    val_cum_loss = 0
                    for _ in range(rep):
                        val_inputs = next(self.vdl).cuda()
                        if self.randomize_cardiac_cycle:
                            self.slab_size = 12 # torch.randint(8, 24, (1,)).item()
                            val_inputs = self.choose_slab(val_inputs)
                        if self.model.module.noise_based:
                            noise = torch.randn_like(val_inputs)
                            val_outputs = self.ema_model(val_inputs, noise=noise)
                            val_loss = torch.mean(self.loss_fn(noise, val_outputs))
                        else:
                            val_outputs = self.ema_model(val_inputs)
                            val_loss = torch.mean(self.loss_fn(val_inputs, val_outputs))
                        val_cum_loss += val_loss
                        
                    if dist.get_rank() == 0:
                        self.writer.add_scalar('Loss/validation', val_cum_loss/rep, self.step)

            if self.step % self.update_ema_every == 0 and dist.get_rank() == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.sample_and_save_every == 0:
                if dist.get_rank() == 0:
                    self.milestone += 1

                    if not self.predictMiddleFrame:
                        x_original = next(self.vdl).cuda()
                        if self.randomize_cardiac_cycle:
                            self.slab_size = 12
                            x_original = self.choose_slab(x_original)
                        b, c, f, h, w = x_original.shape

                        # degrade the images and sample from the EMA model
                        t = torch.tensor([self.model.module.timesteps - 1], device=x_original.device)
                        x_degraded = self.model.module.degrade(x=x_original, t=t) 
                        x_direct, x_recon = self.ema_model.module.sample(x=x_degraded, t=t, eta=self.eta)

                        if c == 2:
                            # mag_original = abs(torch.complex(x_original[:, 0], x_original[:, 1])).unsqueeze(1)
                            mag_recon = abs(torch.complex(x_recon[:, 0], x_recon[:, 1])).unsqueeze(1)
                            mag_direct = abs(torch.complex(x_direct[:, 0], x_direct[:, 1])).unsqueeze(1)
                            mag_degraded = abs(torch.complex(x_degraded[:, 0], x_degraded[:, 1])).unsqueeze(1)
                        elif c == 1:
                            # mag_original = x_original
                            mag_recon = x_recon
                            mag_direct = x_direct
                            mag_degraded = x_degraded
                        else:
                            raise NotImplementedError("Image dimensions not supported. Default: 2 channels")

                        # mag_original = rearrange(mag_original, 'b c f h w -> (b f) c h w')
                        # mag_original = normalize(mag_original)
                        # self.writer.add_images('Sample/original', mag_original, self.step)

                        mag_recon = rearrange(mag_recon, 'b c f h w -> (b f) c h w')
                        mag_recon = normalize(mag_recon)
                        self.writer.add_images('Sample/recon', mag_recon, self.step)

                        mag_direct = rearrange(mag_direct, 'b c f h w -> (b f) c h w')
                        mag_direct = normalize(mag_direct)
                        self.writer.add_images('Sample/direct', mag_direct, self.step)

                        mag_degraded = rearrange(mag_degraded, 'b c f h w -> (b f) c h w')
                        mag_degraded = normalize(mag_degraded)
                        self.writer.add_images('Sample/degraded', mag_degraded, self.step)

                    acc_loss = acc_loss / (self.sample_and_save_every + 1)
                    print(f'Mean of last {self.step}: {acc_loss}')
                    acc_loss = 0

                    self.save(path=self.model_dir, identifier=f'{self.milestone:03}-')
                    torch.cuda.synchronize()

                dist.barrier()
            self.step += 1

        self.writer.flush()
        print('training completed')
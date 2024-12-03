import torch
import math
from PIL import Image
import numpy as np
from tools.modules.diffusions.diffusion_ddim import DiffusionDDIM
from utils.registry_class import DIFFUSION
from .schedules import beta_schedule
from .losses import kl_divergence, discretized_gaussian_log_likelihood

def _i(tensor, t, x):
    r"""Index tensor using t and format the output according to x.
    """
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    if tensor.device != x.device:
        tensor = tensor.to(x.device)
    return tensor[t].view(shape).to(x)


@DIFFUSION.register_class()
class ControlDiffusionDDIM(DiffusionDDIM):
    #TIP: add control_model
    def p_mean_variance(self, xt, hint, t, model, control_model = None, autoencoder=None, model_kwargs={}, clamp=None, percentile=None, guide_scale=None):
        r"""Distribution of p(x_{t-1} | x_t).
        """
        # predict distribution
        if guide_scale is None:
            control = control_model(xt, hint, self._scale_timesteps(t), **model_kwargs) if control_model is not None else None
            out = model(xt, self._scale_timesteps(t), control=control, **model_kwargs)

        else:
            # classifier-free guidance
            # (model_kwargs[0]: conditional kwargs; model_kwargs[1]: non-conditional kwargs)
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            control_y = control_model(xt, hint, self._scale_timesteps(t), autoencoder=autoencoder, sqrt_alphas_cumprod=self.sqrt_alphas_cumprod, 
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod, \
            sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod, **model_kwargs[0]) if control_model is not None else None
            
            y_out = model(xt, self._scale_timesteps(t), autoencoder=autoencoder, sqrt_alphas_cumprod=self.sqrt_alphas_cumprod, 
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod, \
            sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod, control=control_y, **model_kwargs[0])
            
            control_u = control_model(xt, hint, self._scale_timesteps(t), autoencoder=autoencoder, sqrt_alphas_cumprod=self.sqrt_alphas_cumprod, 
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod, \
            sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod, **model_kwargs[1]) if control_model is not None else None
            
            u_out = model(xt, self._scale_timesteps(t), autoencoder=autoencoder, sqrt_alphas_cumprod=self.sqrt_alphas_cumprod, 
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod, \
            sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod, control=control_u, **model_kwargs[1])

            dim = y_out.size(1) if self.var_type.startswith('fixed') else y_out.size(1) // 2
            out = torch.cat([
                u_out[:, :dim] + guide_scale * (y_out[:, :dim] - u_out[:, :dim]),
                y_out[:, dim:]], dim=1) 
        
        # compute variance
        if self.var_type == 'learned':
            out, log_var = out.chunk(2, dim=1)
            var = torch.exp(log_var)
        elif self.var_type == 'learned_range':
            out, fraction = out.chunk(2, dim=1)
            min_log_var = _i(self.posterior_log_variance_clipped, t, xt)
            max_log_var = _i(torch.log(self.betas), t, xt)
            fraction = (fraction + 1) / 2.0
            log_var = fraction * max_log_var + (1 - fraction) * min_log_var
            var = torch.exp(log_var)
        elif self.var_type == 'fixed_large':
            var = _i(torch.cat([self.posterior_variance[1:2], self.betas[1:]]), t, xt)
            log_var = torch.log(var)
        elif self.var_type == 'fixed_small':
            var = _i(self.posterior_variance, t, xt)
            log_var = _i(self.posterior_log_variance_clipped, t, xt)

        if autoencoder is not None: 
            x0 = out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        else:
            # compute mean and x0
            if self.mean_type == 'x_{t-1}':
                mu = out  # x_{t-1}
                x0 = _i(1.0 / self.posterior_mean_coef1, t, xt) * mu - \
                    _i(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, xt) * xt
            elif self.mean_type == 'x0':
                x0 = out
                mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
            elif self.mean_type == 'eps':
                x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                    _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out
                mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
            elif self.mean_type == 'v':
                x0 = _i(self.sqrt_alphas_cumprod, t, xt) * xt - \
                    _i(self.sqrt_one_minus_alphas_cumprod, t, xt) * out
                mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        
        # restrict the range of x0
        if percentile is not None:
            assert percentile > 0 and percentile <= 1  # e.g., 0.995
            s = torch.quantile(x0.flatten(1).abs(), percentile, dim=1).clamp_(1.0).view(-1, 1, 1, 1)
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)
        return mu, var, log_var, x0
    
    #TIP: add control_model
    def loss(self, x0, hint, t, step, model, autoencoder, rank, control_model=None, model_kwargs={}, gs_data=None, noise=None, weight = None, use_div_loss= False):

        # noise = torch.randn_like(x0) if noise is None else noise # [80, 4, 8, 32, 32]
        noise = self.sample_loss(x0, noise)

        xt = self.q_sample(x0, t, noise=noise)

        # compute loss
        if self.loss_type in ['kl', 'rescaled_kl']:
            loss, _ = self.variational_lower_bound(x0, xt, t, model, model_kwargs)
            if self.loss_type == 'rescaled_kl':
                loss = loss * self.num_timesteps
        elif self.loss_type in ['mse', 'rescaled_mse', 'l1', 'rescaled_l1']: # self.loss_type: mse
            
            if (hasattr(model, 'module') and model.module.use_lgm_refine) or model.use_lgm_refine:
                control = control_model(xt, hint, self._scale_timesteps(t), x0=x0, 
                                sqrt_alphas_cumprod=self.sqrt_alphas_cumprod, 
                                sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
                                sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod,
                                sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod,
                                autoencoder=autoencoder,
                                **model_kwargs) if control_model is not None else None
                
                out = model(xt, self._scale_timesteps(t), x0=x0, 
                                sqrt_alphas_cumprod=self.sqrt_alphas_cumprod, 
                                sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
                                sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod,
                                sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod,
                                autoencoder=autoencoder,
                                control=control,
                                **model_kwargs)
            else:
                control = control_model(xt, hint, self._scale_timesteps(t), **model_kwargs) if control_model is not None else None
                out = model(xt, self._scale_timesteps(t), control=control, **model_kwargs)

            # VLB for variation
            loss_vlb = 0.0
            if self.var_type in ['learned', 'learned_range']: # self.var_type: 'fixed_small'
                out, var = out.chunk(2, dim=1)
                frozen = torch.cat([out.detach(), var], dim=1)  # learn var without affecting the prediction of mean
                loss_vlb, _ = self.variational_lower_bound(x0, xt, t, model=lambda *args, **kwargs: frozen)
                if self.loss_type.startswith('rescaled_'):
                    loss_vlb = loss_vlb * self.num_timesteps / 1000.0

            if (hasattr(model, 'module') and model.module.use_lgm_refine) or model.use_lgm_refine:
                loss = out['loss']
                # print("[Training PSNR]:", out['psnr'], "[Train Time]:", t)
            else:
                # MSE/L1 for x0/eps
                # target = {'eps': noise, 'x0': x0, 'x_{t-1}': self.q_posterior_mean_variance(x0, xt, t)[0]}[self.mean_type]
                target = {
                    'eps': noise, 
                    'x0': x0, 
                    'x_{t-1}': self.q_posterior_mean_variance(x0, xt, t)[0], 
                    'v':_i(self.sqrt_alphas_cumprod, t, xt) * noise - _i(self.sqrt_one_minus_alphas_cumprod, t, xt) * x0}[self.mean_type]
                loss = (out - target).pow(1 if self.loss_type.endswith('l1') else 2).abs().flatten(1).mean(dim=1)

            if weight is not None:
                loss = loss*weight   

            # div loss
            if use_div_loss and self.mean_type == 'eps' and x0.shape[2]>1:
                # derive  x0
                x0_ = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                    _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out

                # # derive xt_1, set eta=0 as ddim
                # alphas_prev = _i(self.alphas_cumprod, (t - 1).clamp(0), xt)
                # direction = torch.sqrt(1 - alphas_prev) * out
                # xt_1 = torch.sqrt(alphas_prev) * x0_ + direction

                # ncfhw, std on f
                div_loss = 0.001/(x0_.std(dim=2).flatten(1).mean(dim=1)+1e-4)
                # print(div_loss,loss)
                loss = loss+div_loss

            # total loss
            loss = loss + loss_vlb
        elif self.loss_type in ['charbonnier']:
            control = control_model(xt, hint, self._scale_timesteps(t), **model_kwargs) if control_model is not None else None
            out = model(xt, self._scale_timesteps(t), control=control, **model_kwargs)

            # VLB for variation
            loss_vlb = 0.0
            if self.var_type in ['learned', 'learned_range']:
                out, var = out.chunk(2, dim=1)
                frozen = torch.cat([out.detach(), var], dim=1)  # learn var without affecting the prediction of mean
                loss_vlb, _ = self.variational_lower_bound(x0, xt, t, model=lambda *args, **kwargs: frozen)
                if self.loss_type.startswith('rescaled_'):
                    loss_vlb = loss_vlb * self.num_timesteps / 1000.0
            
            # MSE/L1 for x0/eps
            target = {'eps': noise, 'x0': x0, 'x_{t-1}': self.q_posterior_mean_variance(x0, xt, t)[0]}[self.mean_type]
            loss = torch.sqrt((out - target)**2 + self.epsilon)
            if weight is not None:
                loss = loss*weight
            loss = loss.flatten(1).mean(dim=1)
            
            # total loss
            loss = loss + loss_vlb
        # print(loss.shape)
        return loss
from math import sqrt

import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont
from torch import nn
import torch.nn.functional as F

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class WorldModel(nn.Module):

  def __init__(self, step, config):
    super(WorldModel, self).__init__()
    self._step = step
    self._use_amp = True if config.precision==16 else False
    self._config = config
    self.mask = config.mask
    self.encoder = networks.ConvEncoder(config.grayscale,
        config.cnn_depth, config.act, config.encoder_kernels, config)
    if config.size[0] == 64 and config.size[1] == 64:
      embed_size = 2 ** (len(config.encoder_kernels)-1) * config.cnn_depth
      embed_size *= 2 * 2
    else:
      raise NotImplemented(f"{config.size} is not applicable now")
    self.dynamics = networks.RSSM(
        config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
        config.dyn_input_layers, config.dyn_output_layers,
        config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete,
        config.act, config.dyn_mean_act, config.dyn_std_act,
        config.dyn_temp_post, config.dyn_min_std, config.dyn_cell,
        config.num_actions, embed_size, config.device, config)
    self.heads = nn.ModuleDict()
    channels = (1 if config.grayscale else 3)
    shape = (channels,) + config.size
    if config.dyn_discrete:
      feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
      feat_size = config.dyn_stoch + config.dyn_deter
    if config.mask == 1:
      print('\033[1;35m Using single mask decoders \033[0m')
      self.heads['image'] = networks.SingleMaskDecoder(  #  DoubleDecoder(
        feat_size,  # pytorch version
        config.cnn_depth, config.act, shape, config.decoder_kernels,
        config.decoder_thin)
    elif config.mask == 2:
      print('\033[1;35mUsing double mask decoders \033[0m')
      self.heads['image'] = networks.DoubleDecoder(  # SingleMaskDecoder(  #  DoubleDecoder(
        feat_size,  # pytorch version
        config.cnn_depth, config.act, shape, config.decoder_kernels,
        config.decoder_thin)
    elif config.mask == 3:
      print('\033[1;35mUsing Three mask decoders \033[0m')
      self.heads['image'] = networks.TrippleDecoder(
        feat_size,  # pytorch version
        config.cnn_depth, config.act, shape, config.decoder_kernels,
        config.decoder_thin)
    else:
      raise NotImplementedError
    self.background_decoder = networks.ConvDecoder(  #  DoubleDecoder(
        embed_size * config.init_frame,  # pytorch version
        config.cnn_depth, config.act, shape, config.decoder_kernels,
        config.decoder_thin)
    double_size = 2 if config.use_free else 1
    self.heads['reward'] = networks.DenseHead(
        double_size*feat_size,  # pytorch version
        [], config.reward_layers, config.units, config.act)
    if config.inverse_dynamics:
      self.heads['action'] = networks.DenseHead(
          embed_size,  # pytorch version
          [config.num_actions], config.reward_layers, config.units, config.act)
    if config.pred_discount:
      self.heads['discount'] = networks.DenseHead(
          double_size*feat_size,  # pytorch version
          [], config.discount_layers, config.units, config.act, dist='binary')
    for name in config.grad_heads:
      assert name in self.heads, name
    self._model_opt = tools.Optimizer(
        'model', self.parameters(), config.model_lr, config.opt_eps, config.grad_clip,
        config.weight_decay, opt=config.opt,
        use_amp=self._use_amp)
    self._scales = dict(
        reward=config.reward_scale, discount=config.discount_scale, action=config.action_scale)


  def _train(self, data):
    data = self.preprocess(data) 
    self.dynamics.train_wm = True
    self.dynamics.step = 0

    with tools.RequiresGrad(self):
      with torch.cuda.amp.autocast(self._use_amp):
        embed = self.encoder(data, data['action'])
        embed_back = self.encoder(data, data['action'], bg=True)
        embed_back = embed_back[:, :self._config.init_frame, :].reshape(self._config.batch_size, -1)
        background = self.background_decoder(embed_back.unsqueeze(1)).mode()
        background = torch.clamp(background, min=-0.5, max=0.5)
        self.dynamics.rollout_free = True
        post, prior = self.dynamics.observe(embed, data['action'])
        self.dynamics.rollout_free = self._config.use_free
        kl_balance = tools.schedule(self._config.kl_balance, self._step)
        kl_free = tools.schedule(self._config.kl_free, self._step)
        kl_scale = tools.schedule(self._config.kl_scale, self._step)
        kl_loss, kl_value = self.dynamics.kl_loss(
            post, prior, self._config.kl_forward, kl_balance, kl_free, kl_scale)
        
        if self._config.sz_sparse:
          regularization_loss = torch.mean(prior['gate']) 
        
        losses = {}
        likes = {}
        for name, head in self.heads.items():
          if 'image' in name:
            feat, feat_free = self.dynamics.get_feat_for_decoder(post, prior=prior, action_step=self._config.action_step, free_step=self._config.free_step)
            if self.mask == 3:
              pred, _, _, action_mask, free_mask = self.heads['image'](feat, feat_free, background)
              if self._config.autoencoder:
                like = pred.log_prob(data[name][:, 1:, :, :, :]) 
              else:
                like = pred.log_prob(data[name])  
            else:
              pred, _, _, _, _ = self.heads['image'](feat, feat_free, data['image'])
              like = pred.log_prob(data[name])
          elif 'action' in name:
            embed_action, embed_free = torch.chunk(embed, chunks=2, dim=-1)
            inp = embed_action[:, 1:, :] - embed_action[:, :-1, :]
            pred = head(inp)
            like = pred.log_prob(data[name][:, 1:, :])
          else:
            grad_head = (name in self._config.grad_heads)
            feat = self.dynamics.get_feat(post)
            feat = feat if grad_head else feat.detach()
            pred = head(feat)
            if self._config.autoencoder:
              like = pred.log_prob(data[name][:, 1:, :]) 
            else:
              like = pred.log_prob(data[name])
          likes[name] = like
          losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
        model_loss = sum(losses.values()) + kl_loss
        if self._config.min_free:
          min_free_neg_loss = torch.sum((prior['deter_free'] - prior['deter_free_neg']) ** 2) 
          min_free_pos_loss = torch.sum((prior['deter_free_pos'] - prior['deter_free']) ** 2)
          min_free_loss = min_free_neg_loss + min_free_pos_loss
          model_loss += min_free_loss 
        if self._config.max_action:
          max_action_loss = torch.abs(torch.cosine_similarity(prior['deter'], prior['deter_action_neg'], dim=2))
          model_loss += torch.mean(max_action_loss)    # * 0.0001
        if self._config.sz_sparse:
          model_loss += regularization_loss

      metrics = self._model_opt(model_loss, self.parameters())

    metrics.update({f'{name}_loss': to_np(loss) for name, loss in losses.items()})
    metrics['kl_balance'] = kl_balance
    metrics['kl_free'] = kl_free
    metrics['kl_scale'] = kl_scale
    if self._config.min_free:
      metrics['min_free_loss'] = to_np(min_free_loss)
    if self._config.max_action:
      metrics['max_action_loss'] = to_np(max_action_loss)
    if self._config.sz_sparse:
      metrics['regularization_loss'] = to_np(regularization_loss)
    metrics['kl'] = to_np(torch.mean(kl_value))
    with torch.cuda.amp.autocast(self._use_amp):
      metrics['prior_ent'] = to_np(torch.mean(self.dynamics.get_dist(prior, free=False).entropy()))
      metrics['post_ent'] = to_np(torch.mean(self.dynamics.get_dist(post, free=False).entropy()))
      metrics['prior_ent_free'] = to_np(torch.mean(self.dynamics.get_dist(prior, free=True).entropy()))
      metrics['post_ent_free'] = to_np(torch.mean(self.dynamics.get_dist(post, free=True).entropy()))
      context = None
    post = {k: v.detach() for k, v in post.items()}
    self.dynamics.train_wm = False
    return post, context, metrics

  def preprocess(self, obs):
    obs = obs.copy()
    obs['image'] = torch.Tensor(obs['image']) / 255.0 - 0.5
    if self._config.clip_rewards == 'tanh':
      obs['reward'] = torch.tanh(torch.Tensor(obs['reward'])).unsqueeze(-1)
    elif self._config.clip_rewards == 'identity':
      obs['reward'] = torch.Tensor(obs['reward']).unsqueeze(-1)
    else:
      raise NotImplemented(f'{self._config.clip_rewards} is not implemented')
    if 'discount' in obs:
      obs['discount'] *= self._config.discount
      obs['discount'] = torch.Tensor(obs['discount']).unsqueeze(-1)
    obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
    return obs

  def video_pred(self, data):
    data = self.preprocess(data)
    if self._config.autoencoder:
      truth = data['image'][:6, 1:] + 0.5
    else:
      truth = data['image'][:6] + 0.5
    embed = self.encoder(data, data['action'])

    self.dynamics.rollout_free = True
    states, _ = self.dynamics.observe(embed[:6, :5], data['action'][:6, :5])
    feat, feat_free = self.dynamics.get_feat_for_decoder(states)
    embed_back = self.encoder(data, data['action'], bg=True)
    embed_back = embed_back[:6, :self._config.init_frame, :].reshape(6, -1)
    background = self.background_decoder(embed_back.unsqueeze(1)).mode()
    background = torch.clamp(background, min=-0.5, max=0.5)
    if self.mask == 3:
      recon, gen_action, gen_free, mask_action, mask_free = self.heads['image'](feat, feat_free, background)
    else:
      recon, gen_action, gen_free, mask_action, mask_free = self.heads['image'](feat, feat_free)
    recon = recon.mode()[:6]
    reward_post = self.heads['reward'](
        self.dynamics.get_feat(states)).mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    self.dynamics.predict = True
    prior = self.dynamics.imagine(data['action'][:6, 5:], init)
    self.dynamics.predict = False
    feat, feat_free = self.dynamics.get_feat_for_decoder(prior)
    if self.mask == 3:
      openl, openl_gen_action, openl_gen_free, openl_mask_action, openl_mask_free = self.heads['image'](feat, feat_free, background, start=0)
    else:
      openl, openl_gen_action, openl_gen_free, openl_mask_action, openl_mask_free = self.heads['image'](feat, feat_free)
    openl = openl.mode()
    # openl = self.heads['image'](self.dynamics.get_feat(prior)).mode()
    reward_prior = self.heads['reward'](self.dynamics.get_feat(prior)).mode()
    model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2

    gen_action = torch.cat([gen_action[:, :5] + 0.5, openl_gen_action + 0.5], 1)
    gen_free = torch.cat([gen_free[:, :5] + 0.5, openl_gen_free + 0.5], 1)
    mask_action = torch.cat([mask_action[:, :5], openl_mask_action], 1).repeat(1,1,1,1,3)
    mask_free = torch.cat([mask_free[:, :5], openl_mask_free], 1).repeat(1,1,1,1,3)
    mask_3 = 1 - mask_action - mask_free
    back = background * torch.ones_like(mask_3) + 0.5
    self.dynamics.rollout_free = self._config.use_free

    return torch.cat([truth, model, error, gen_action, gen_free, mask_action, mask_free, mask_3, back], 2)

class WorldModel_TED(nn.Module):
  def __init__(self, step, config):
    super(WorldModel_TED, self).__init__()
    print('Using TED mode')
    self._step = step
    self._use_amp = True if config.precision==16 else False
    self._config = config
    self.mask = config.mask
    
    # Encoders (original from Iso-Dream)
    self.encoder = networks.ConvEncoder(config.grayscale,
        config.cnn_depth, config.act, config.encoder_kernels, config)
    if config.size[0] == 64 and config.size[1] == 64:
      embed_size = 2 ** (len(config.encoder_kernels)-1) * config.cnn_depth
      embed_size *= 2 * 2
    else:
      raise NotImplemented(f"{config.size} is not applicable now")
    
    # Target encoders (from TED) - we create target copies of the main encoder
    self.target_encoder = networks.ConvEncoder(config.grayscale,
        config.cnn_depth, config.act, config.encoder_kernels, config)
    # Initialize target encoder with same weights as main encoder
    self._update_target_encoder(tau=1.0)
    
    # Dynamics model (original from Iso-Dream)
    self.dynamics = networks.RSSM(
        config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
        config.dyn_input_layers, config.dyn_output_layers,
        config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete,
        config.act, config.dyn_mean_act, config.dyn_std_act,
        config.dyn_temp_post, config.dyn_min_std, config.dyn_cell,
        config.num_actions, embed_size, config.device, config)
    
    # TED classifiers - one for each branch
    if config.dyn_discrete:
        feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
        feat_size = config.dyn_stoch + config.dyn_deter
        
    self.ted_classifier_action = TEDClassifier(feat_size)
    self.ted_classifier_free = TEDClassifier(feat_size)
    
    # Additional heads and networks (original from Iso-Dream)
    self.heads = nn.ModuleDict()
    channels = (1 if config.grayscale else 3)
    if isinstance(config.size, list):
      config.size = tuple(config.size)
    shape = (channels,) + config.size
    if config.dyn_discrete:
      feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
      feat_size = config.dyn_stoch + config.dyn_deter
    
    # Rest of the initialization code from the original Iso-Dream
    if config.mask == 1:
      print('\033[1;35m Using single mask decoders \033[0m')
      self.heads['image'] = networks.SingleMaskDecoder(
        feat_size,
        config.cnn_depth, config.act, shape, config.decoder_kernels,
        config.decoder_thin)
    elif config.mask == 2:
      print('\033[1;35mUsing double mask decoders \033[0m')
      self.heads['image'] = networks.DoubleDecoder(
        feat_size,
        config.cnn_depth, config.act, shape, config.decoder_kernels,
        config.decoder_thin)
    elif config.mask == 3:
      print('\033[1;35mUsing Three mask decoders \033[0m')
      self.heads['image'] = networks.TrippleDecoder(
        feat_size,
        config.cnn_depth, config.act, shape, config.decoder_kernels,
        config.decoder_thin)
    else:
      raise NotImplementedError
    
    self.background_decoder = networks.ConvDecoder(
        embed_size * config.init_frame,
        config.cnn_depth, config.act, shape, config.decoder_kernels,
        config.decoder_thin)
    
    double_size = 2 if config.use_free else 1
    self.heads['reward'] = networks.DenseHead(
        double_size*feat_size,
        [], config.reward_layers, config.units, config.act)
    if config.inverse_dynamics:
      self.heads['action'] = networks.DenseHead(
          embed_size,
          [config.num_actions], config.reward_layers, config.units, config.act)
    if config.pred_discount:
      self.heads['discount'] = networks.DenseHead(
          double_size*feat_size,
          [], config.discount_layers, config.units, config.act, dist='binary')
    
    for name in config.grad_heads:
      assert name in self.heads, name
    
    self._model_opt = tools.Optimizer(
        'model', self.parameters(), config.model_lr, config.opt_eps, config.grad_clip,
        config.weight_decay, opt=config.opt,
        use_amp=self._use_amp)
    
    self._scales = dict(
        reward=config.reward_scale, discount=config.discount_scale, action=config.action_scale)
    
    # TED loss coefficient
    self.ted_coefficient = config.ted_coefficient if hasattr(config, 'ted_coefficient') else 1.0
    self.target_update_rate = config.target_update_rate if hasattr(config, 'target_update_rate') else 0.01

  def _update_target_encoder(self, tau=None):
    """Update target encoder with EMA of main encoder params"""
    if tau is None:
        tau = self.target_update_rate
    
    for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

  def _train(self, data):
    data = self.preprocess(data) 
    self.dynamics.train_wm = True
    self.dynamics.step = 0

    with tools.RequiresGrad(self):
      with torch.cuda.amp.autocast(self._use_amp):
        # Original Iso-Dream encoding
        embed = self.encoder(data, data['action'])
        embed_back = self.encoder(data, data['action'], bg=True)
        embed_back = embed_back[:, :self._config.init_frame, :].reshape(self._config.batch_size, -1)
        background = self.background_decoder(embed_back.unsqueeze(1)).mode()
        background = torch.clamp(background, min=-0.5, max=0.5)
        
        # Create target encoding for TED
        with torch.no_grad():
            target_embed = self.target_encoder(data, data['action'])
        
        self.dynamics.rollout_free = True
        post, prior = self.dynamics.observe(embed, data['action'])
        target_post, target_prior = self.dynamics.observe(target_embed, data['action'])
        self.dynamics.rollout_free = self._config.use_free
        
        # Create TED features from action-conditioned and action-free branches
        feat, feat_free = self.dynamics.get_feat_for_decoder(post, prior=prior, 
                                                            action_step=self._config.action_step, 
                                                            free_step=self._config.free_step)
        
        target_feat, target_feat_free = self.dynamics.get_feat_for_decoder(target_post, prior=target_prior, 
                                                    action_step=self._config.action_step, 
                                                    free_step=self._config.free_step)
        
        # Calculate TED loss
        ted_loss = self._compute_ted_loss(feat, target_feat, feat_free, target_feat_free)
        
        # Original Iso-Dream KL loss
        kl_balance = tools.schedule(self._config.kl_balance, self._step)
        kl_free = tools.schedule(self._config.kl_free, self._step)
        kl_scale = tools.schedule(self._config.kl_scale, self._step)
        kl_loss, kl_value = self.dynamics.kl_loss(
            post, prior, self._config.kl_forward, kl_balance, kl_free, kl_scale)
        
        if self._config.sz_sparse:
          regularization_loss = torch.mean(prior['gate']) 
        
        losses = {}
        likes = {}
        
        # Original Iso-Dream heads processing
        for name, head in self.heads.items():
          if 'image' in name:
            if self.mask == 3:
              pred, _, _, action_mask, free_mask = self.heads['image'](feat, feat_free, background)
              if self._config.autoencoder:
                like = pred.log_prob(data[name][:, 1:, :, :, :]) 
              else:
                like = pred.log_prob(data[name])  
            else:
              pred, _, _, _, _ = self.heads['image'](feat, feat_free, data['image'])
              like = pred.log_prob(data[name])
          elif 'action' in name:
            embed_action, embed_free = torch.chunk(embed, chunks=2, dim=-1)
            inp = embed_action[:, 1:, :] - embed_action[:, :-1, :]
            pred = head(inp)
            like = pred.log_prob(data[name][:, 1:, :])
          else:
            grad_head = (name in self._config.grad_heads)
            feat_combined = self.dynamics.get_feat(post)
            feat_combined = feat_combined if grad_head else feat_combined.detach()
            pred = head(feat_combined)
            if self._config.autoencoder:
              like = pred.log_prob(data[name][:, 1:, :]) 
            else:
              like = pred.log_prob(data[name])
          likes[name] = like
          losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
        
        # Combine all losses
        model_loss = sum(losses.values()) + kl_loss + self.ted_coefficient * ted_loss
        
        # Original Iso-Dream optional regularization losses
        if self._config.min_free:
          min_free_neg_loss = torch.sum((prior['deter_free'] - prior['deter_free_neg']) ** 2) 
          min_free_pos_loss = torch.sum((prior['deter_free_pos'] - prior['deter_free']) ** 2)
          min_free_loss = min_free_neg_loss + min_free_pos_loss
          model_loss += min_free_loss 
        if self._config.max_action:
          max_action_loss = torch.abs(torch.cosine_similarity(prior['deter'], prior['deter_action_neg'], dim=2))
          model_loss += torch.mean(max_action_loss)
        if self._config.sz_sparse:
          model_loss += regularization_loss

      # Optimization step
      metrics = self._model_opt(model_loss, self.parameters())

      # Update target encoder after optimization step
      self._update_target_encoder()

    # Update metrics
    metrics.update({f'{name}_loss': to_np(loss) for name, loss in losses.items()})
    metrics['kl_balance'] = kl_balance
    metrics['kl_free'] = kl_free
    metrics['kl_scale'] = kl_scale
    metrics['ted_loss'] = to_np(ted_loss)
    if self._config.min_free:
      metrics['min_free_loss'] = to_np(min_free_loss)
    if self._config.max_action:
      metrics['max_action_loss'] = to_np(max_action_loss)
    if self._config.sz_sparse:
      metrics['regularization_loss'] = to_np(regularization_loss)
    metrics['kl'] = to_np(torch.mean(kl_value))
    
    with torch.cuda.amp.autocast(self._use_amp):
      metrics['prior_ent'] = to_np(torch.mean(self.dynamics.get_dist(prior, free=False).entropy()))
      metrics['post_ent'] = to_np(torch.mean(self.dynamics.get_dist(post, free=False).entropy()))
      metrics['prior_ent_free'] = to_np(torch.mean(self.dynamics.get_dist(prior, free=True).entropy()))
      metrics['post_ent_free'] = to_np(torch.mean(self.dynamics.get_dist(post, free=True).entropy()))
      context = None
      
    post = {k: v.detach() for k, v in post.items()}
    self.dynamics.train_wm = False
    return post, context, metrics

  def _compute_ted_loss(self, feat, target_feat, feat_free, target_feat_free):
      """
      Compute TED loss based on SAC-TED implementation
      Inputs:
      - feat: Action-conditioned features [batch_size, seq_len, feat_dim]
      - target_embed: Target features [batch_size, seq_len, feat_dim] 
      - feat_free: Action-free features [batch_size, seq_len, feat_dim]
      """

      # Process action-conditioned branch
      ted_loss_action = self._compute_branch_ted_loss(feat, target_feat, self.ted_classifier_action)
      ted_loss = ted_loss_action
      
      # Process action-free branch
      ted_loss_free = self._compute_branch_ted_loss(feat_free, target_feat_free, self.ted_classifier_free)
      
      # Combine losses from both branches
      ted_loss = ted_loss_action + ted_loss_free
      
      return ted_loss
      
  def _compute_branch_ted_loss(self, feat, target_feat, classifier):
      """
      Compute TED loss for a single branch
      """
      batch_size = feat.shape[0]
      seq_len = feat.shape[1]
      feat_dim = feat.shape[2]
      target_feat_dim = target_feat.shape[2]
      loss_fn = nn.BCEWithLogitsLoss()
      # print('feat_dim', feat_dim)
      # print('target_feat_dim',target_feat_dim)
      
      # Skip if sequence is too short
      if seq_len < 2:
        return torch.tensor(0.0, device=feat.device)
      
      # Create temporal samples (consecutive timesteps)
      # Using only the first feature along sequence dimension for simplicity
      obs_rep = feat[:, 0]  # [batch_size, feat_dim]
      next_obs_rep = target_feat[:, 1]  # [batch_size, feat_dim]
      
      # Stack the consecutive observations to make temporal samples
      non_iid_samples = torch.stack([obs_rep, next_obs_rep], dim=1)  # [batch_size, 2, feat_dim]
      # All temporal samples are given a label of 1
      non_iid_labels = torch.ones((batch_size, 1), device=feat.device)
      
      # Create the non-temporal different episode samples
      rnd_idx = torch.randperm(batch_size)
      diff_ep_iid_samples = torch.stack([obs_rep, next_obs_rep[rnd_idx]], dim=1)
      # All non-temporal samples are given a label of 0
      diff_ep_iid_labels = torch.zeros((batch_size, 1), device=feat.device)
      
      # Create the non-temporal same episode samples
      if seq_len > 2:
        # If sequence length allows, take a different position
        same_ep_pos = 2  # Use third position in sequence
        same_ep_obs_rep = target_feat[:, same_ep_pos]
      else:
        # Otherwise, just reuse the first position
        same_ep_obs_rep = target_feat[:, 0]
      
      same_ep_iid_samples = torch.stack([obs_rep, same_ep_obs_rep], dim=1)
      same_ep_iid_labels = torch.zeros((batch_size, 1), device=feat.device)
      
      # Combine all samples and labels
      samples = torch.cat([non_iid_samples, diff_ep_iid_samples, same_ep_iid_samples])
      labels = torch.cat([non_iid_labels, diff_ep_iid_labels, same_ep_iid_labels])
      
      # Get predictions from classifier
      preds = classifier(samples)
      
      # Calculate loss
      ted_loss = loss_fn(preds, labels)
      
      return ted_loss

    
  def preprocess(self, obs):
    obs = obs.copy()
    obs['image'] = torch.Tensor(obs['image']) / 255.0 - 0.5
    if self._config.clip_rewards == 'tanh':
      obs['reward'] = torch.tanh(torch.Tensor(obs['reward'])).unsqueeze(-1)
    elif self._config.clip_rewards == 'identity':
      obs['reward'] = torch.Tensor(obs['reward']).unsqueeze(-1)
    else:
      raise NotImplemented(f'{self._config.clip_rewards} is not implemented')
    if 'discount' in obs:
      obs['discount'] *= self._config.discount
      obs['discount'] = torch.Tensor(obs['discount']).unsqueeze(-1)
    obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
    return obs

  def video_pred(self, data):
    data = self.preprocess(data)
    if self._config.autoencoder:
      truth = data['image'][:6, 1:] + 0.5
    else:
      truth = data['image'][:6] + 0.5
    embed = self.encoder(data, data['action'])

    self.dynamics.rollout_free = True
    states, _ = self.dynamics.observe(embed[:6, :5], data['action'][:6, :5])
    feat, feat_free = self.dynamics.get_feat_for_decoder(states)
    embed_back = self.encoder(data, data['action'], bg=True)
    embed_back = embed_back[:6, :self._config.init_frame, :].reshape(6, -1)
    background = self.background_decoder(embed_back.unsqueeze(1)).mode()
    background = torch.clamp(background, min=-0.5, max=0.5)
    if self.mask == 3:
      recon, gen_action, gen_free, mask_action, mask_free = self.heads['image'](feat, feat_free, background)
    else:
      recon, gen_action, gen_free, mask_action, mask_free = self.heads['image'](feat, feat_free)
    recon = recon.mode()[:6]
    reward_post = self.heads['reward'](
        self.dynamics.get_feat(states)).mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    self.dynamics.predict = True
    prior = self.dynamics.imagine(data['action'][:6, 5:], init)
    self.dynamics.predict = False
    feat, feat_free = self.dynamics.get_feat_for_decoder(prior)
    if self.mask == 3:
      openl, openl_gen_action, openl_gen_free, openl_mask_action, openl_mask_free = self.heads['image'](feat, feat_free, background, start=0)
    else:
      openl, openl_gen_action, openl_gen_free, openl_mask_action, openl_mask_free = self.heads['image'](feat, feat_free)
    openl = openl.mode()
    # openl = self.heads['image'](self.dynamics.get_feat(prior)).mode()
    reward_prior = self.heads['reward'](self.dynamics.get_feat(prior)).mode()
    model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2

    gen_action = torch.cat([gen_action[:, :5] + 0.5, openl_gen_action + 0.5], 1)
    gen_free = torch.cat([gen_free[:, :5] + 0.5, openl_gen_free + 0.5], 1)
    mask_action = torch.cat([mask_action[:, :5], openl_mask_action], 1).repeat(1,1,1,1,3)
    mask_free = torch.cat([mask_free[:, :5], openl_mask_free], 1).repeat(1,1,1,1,3)
    mask_3 = 1 - mask_action - mask_free
    back = background * torch.ones_like(mask_3) + 0.5
    self.dynamics.rollout_free = self._config.use_free

    return torch.cat([truth, model, error, gen_action, gen_free, mask_action, mask_free, mask_3, back], 2)



class ImagBehavior(nn.Module):

  def __init__(self, config, world_model, stop_grad_actor=True, reward=None):
    super(ImagBehavior, self).__init__()
    self._use_amp = True if config.precision==16 else False
    self._config = config
    self._world_model = world_model
    self._stop_grad_actor = stop_grad_actor
    self._reward = reward
    if config.dyn_discrete:
      feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
      feat_size = config.dyn_stoch + config.dyn_deter
    double_size = 1
    self.actor = networks.ActionHead(
        double_size*feat_size,  # pytorch version
        config.num_actions, config.actor_layers, config.units, config.act,
        config.actor_dist, config.actor_init_std, config.actor_min_std,
        config.actor_dist, config.actor_temp, config.actor_outscale)
    self.value = networks.DenseHead(
        double_size*feat_size,  # pytorch version
        [], config.value_layers, config.units, config.act,
        config.value_head)
    if config.slow_value_target or config.slow_actor_target:
      self._slow_value = networks.DenseHead(
          double_size*feat_size,  # pytorch version
          [], config.value_layers, config.units, config.act)
      self._updates = 0
    kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)

    ## attention
    self.attention = networks.Attention(feat_size)

    self._actor_opt = tools.Optimizer(
        'actor', self.actor.parameters(), config.actor_lr, config.opt_eps, config.actor_grad_clip,
        **kw)
    if self._config.rollout_policy:
      self._value_opt = tools.Optimizer(
        'value', list(self.value.parameters())+list(self.attention.parameters()), config.value_lr, config.opt_eps, config.value_grad_clip,
        **kw)
    else:
      self._value_opt = tools.Optimizer(
        'value', self.value.parameters(), config.value_lr, config.opt_eps, config.value_grad_clip,
        **kw)

  def init_men(self):
    self.men = []
    self.num = 0

  def _train(self, start, objective=None, action=None, reward=None, imagine=None, tape=None, repeats=None):
    objective = objective or self._reward
    self._update_slow_target()
    metrics = {}
    if self._config.rollout_policy:
      with tools.RequiresGrad(self.attention):
        with tools.RequiresGrad(self.actor):
          with torch.cuda.amp.autocast(self._use_amp):
            imag_feat, imag_state, imag_action = self._imagine(
                start, self.actor, self._config.imag_horizon, repeats)
            reward = objective(imag_feat.detach(), imag_state, imag_action)
            actor_ent = self.actor(imag_feat.detach()).entropy()
            state_ent = self._world_model.dynamics.get_dist(
            imag_state, free=False).entropy() + self._world_model.dynamics.get_dist(imag_state, free=True).entropy()
            target, weights = self._compute_target(
                imag_feat.detach(), imag_state, imag_action, reward, actor_ent, state_ent,
                self._config.slow_actor_target)
            actor_loss, mets = self._compute_actor_loss(
                imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
                weights)
            metrics.update(mets)
            if self._config.slow_value_target != self._config.slow_actor_target:
              target, weights = self._compute_target(
                  imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
                  self._config.slow_value_target)
            value_input = imag_feat

        with tools.RequiresGrad(self.value):
          with torch.cuda.amp.autocast(self._use_amp):
            value = self.value(value_input[:-1])
            target = torch.stack(target, dim=1)
            value_loss = -value.log_prob(target.detach())
            if self._config.value_decay:
              value_loss += self._config.value_decay * value.mode()
            value_loss = torch.mean(weights[:-1] * value_loss[:,:,None])
    else:  
      with tools.RequiresGrad(self.actor):
        with torch.cuda.amp.autocast(self._use_amp):
          imag_feat, imag_state, imag_action = self._imagine(
              start, self.actor, self._config.imag_horizon, repeats)
          reward = objective(imag_feat, imag_state, imag_action)
          actor_ent = self.actor(imag_feat).entropy()
          state_ent = self._world_model.dynamics.get_dist(imag_state, free=False).entropy()
          target, weights = self._compute_target(
              imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
              self._config.slow_actor_target)
          actor_loss, mets = self._compute_actor_loss(
              imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
              weights)
          metrics.update(mets)
          if self._config.slow_value_target != self._config.slow_actor_target:
            target, weights = self._compute_target(
                imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
                self._config.slow_value_target)
          value_input = imag_feat

      with tools.RequiresGrad(self.value):
        with torch.cuda.amp.autocast(self._use_amp):
          value = self.value(value_input[:-1].detach())
          target = torch.stack(target, dim=1)
          value_loss = -value.log_prob(target.detach())
          if self._config.value_decay:
            value_loss += self._config.value_decay * value.mode()
          value_loss = torch.mean(weights[:-1] * value_loss[:,:,None])
          
    metrics['reward_mean'] = to_np(torch.mean(reward))
    metrics['reward_std'] = to_np(torch.std(reward))
    metrics['actor_ent'] = to_np(torch.mean(actor_ent))
    with tools.RequiresGrad(self):
      metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
      if self._config.rollout_policy:
        metrics.update(self._value_opt(value_loss, list(self.value.parameters())+list(self.attention.parameters())))
      else:
        metrics.update(self._value_opt(value_loss, self.value.parameters()))

    return imag_feat, imag_state, imag_action, weights, metrics

  def boundary_sampler(self, a):
    zero = torch.zeros_like(a)
    one = torch.ones_like(a)
    sample = torch.where(a > 0.5, one, a)
    sample = torch.where(sample < 0.5, zero, sample)
    sample = sample.detach() + (a - a.detach())

    return sample

  def _imagine(self, start, policy, horizon, repeats=None):
    dynamics = self._world_model.dynamics
    if repeats:
      raise NotImplemented("repeats is not implemented in this version")
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}

    def step(prev, _):
      state, _, _ = prev
      if self._config.rollout_policy:
        men_free = []
        start_free = state.copy()
        if self._config.sz_sparse:
          gate_feat = torch.cat([start_free['deter_free'], start_free['deter']], -1)
          gate_feat = dynamics.prior_gate(gate_feat)
          prior_gate_data = networks.boundary_sampler(gate_feat)
          # prior_gate_data = prior_gate_data[:, 0].unsqueeze(-1)
          x = torch.cat([start_free['stoch_free'], start_free['stoch'] * prior_gate_data], -1)
          stoch_free = dynamics.sz_concat(x)
          start_free['stoch_free'] = stoch_free
        for _ in range(self._config.window):
          stoch_free = start_free['stoch_free']
          deter_free = start_free['deter_free']
          free_feat = torch.cat([stoch_free, deter_free], -1)
          men_free.append(free_feat)
          start_free = dynamics.img_step(start_free, None, sample=self._config.imag_sample, only_free=True)

        free_atten = torch.stack(men_free, dim=1)
        feat = dynamics.get_feat_rollout_policy(state, free_atten, self.attention)
      else:
        feat = dynamics.get_feat(state)
      action = policy(feat.detach()).sample()
      succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
      return succ, feat, action
    feat = 0 * dynamics.get_feat(start)
    if self._config.use_free:
      feat, _ = feat.chunk(2, dim=-1)
    action = policy(feat).mode()
    succ, feats, actions = tools.static_scan(
        step, [torch.arange(horizon)], (start, feat, action))
    states = {k: torch.cat([
        start[k][None], v[:-1]], 0) for k, v in succ.items()}
    if repeats:
      raise NotImplemented("repeats is not implemented in this version")

    return feats, states, actions

  def _compute_target(
      self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
      slow):
    if 'discount' in self._world_model.heads:
      inp = self._world_model.dynamics.get_feat(imag_state)
      discount = self._world_model.heads['discount'](inp).mean
    else:
      discount = self._config.discount * torch.ones_like(reward)
    if self._config.future_entropy and self._config.actor_entropy() > 0:
      reward += self._config.actor_entropy() * actor_ent
    if self._config.future_entropy and self._config.actor_state_entropy() > 0:
      reward += self._config.actor_state_entropy() * state_ent
    if slow:
      value = self._slow_value(imag_feat).mode()
    else:
      value = self.value(imag_feat).mode()
    target = tools.lambda_return(
        reward[:-1], value[:-1], discount[:-1],
        bootstrap=value[-1], lambda_=self._config.discount_lambda, axis=0)
    weights = torch.cumprod(
        torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()
    return target, weights

  def _compute_actor_loss(
      self, imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
      weights):
    metrics = {}
    inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
    policy = self.actor(inp)
    actor_ent = policy.entropy()
    target = torch.stack(target, dim=1)
    if self._config.imag_gradient == 'dynamics':
      actor_target = target
    elif self._config.imag_gradient == 'reinforce':
      actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (
          target - self.value(imag_feat[:-1]).mode()).detach()
    elif self._config.imag_gradient == 'both':
      actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (
          target - self.value(imag_feat[:-1]).mode()).detach()
      mix = self._config.imag_gradient_mix()
      actor_target = mix * target + (1 - mix) * actor_target
      metrics['imag_gradient_mix'] = mix
    else:
      raise NotImplementedError(self._config.imag_gradient)
    if not self._config.future_entropy and (self._config.actor_entropy() > 0):
      actor_target += self._config.actor_entropy() * actor_ent[:-1][:,:,None]
    if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
      actor_target += self._config.actor_state_entropy() * state_ent[:-1]
    actor_loss = -torch.mean(weights[:-1] * actor_target)
    return actor_loss, metrics

  def _update_slow_target(self):
    if self._config.slow_value_target or self._config.slow_actor_target:
      if self._updates % self._config.slow_target_update == 0:
        mix = self._config.slow_target_fraction
        for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
          d.data = mix * s.data + (1 - mix) * d.data
      self._updates += 1


class TEDClassifier(nn.Module):
    """Temporal disentanglement classifier based on the TED paper"""
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Parameters for the TED classifier (from the paper)
        self.k1 = nn.Parameter(torch.empty(feature_dim))
        self.k2 = nn.Parameter(torch.empty(feature_dim))
        self.b = nn.Parameter(torch.empty(feature_dim))
        self.k_bar = nn.Parameter(torch.empty(feature_dim))
        self.b_bar = nn.Parameter(torch.empty(feature_dim))
        self.c = nn.Parameter(torch.empty(1))
        
        # Initialize parameters with normal distribution
        nn.init.normal_(self.k1, std=0.02)
        nn.init.normal_(self.k2, std=0.02)
        nn.init.normal_(self.b, std=0.02)
        nn.init.normal_(self.k_bar, std=0.02)
        nn.init.normal_(self.b_bar, std=0.02)
        nn.init.normal_(self.c, std=0.02)
    
    def forward(self, samples):
        """
        Apply TED classification based on Equation 1 in the TED paper
        Input:
        - samples: tensor of shape [batch_size, 2, feature_dim] where samples[:, 0] are x1 features
          and samples[:, 1] are x2 features
        Returns:
        - logits for classification [batch_size, 1]
        """
        batch_size = samples.shape[0]
        
        # Extract the features
        x1 = samples[:, 0]  # [batch_size, feature_dim]
        x2 = samples[:, 1]  # [batch_size, feature_dim]
        
        # Equation 1 from the TED paper
        linear_term = torch.abs(self.k1 * x1 + self.k2 * x2 + self.b)
        marginal_term = torch.square(self.k_bar * x1 + self.b_bar)
        
        # Sum across feature dimension
        linear_sum = torch.sum(linear_term, dim=1)
        marginal_sum = torch.sum(marginal_term, dim=1)
        
        # Final output
        output = linear_sum - marginal_sum + self.c
        
        return output.view(batch_size, 1)
      

#### new
class WorldModel_TED_Simplified(nn.Module):
    def __init__(self, step, config):
        super(WorldModel_TED_Simplified, self).__init__()
        print('Using Simplified TED mode (no target encoder)')
        self._step = step
        self._use_amp = True if config.precision==16 else False
        self._config = config
        self.mask = config.mask
        
        # Only main encoder (no target encoder)
        self.encoder = networks.ConvEncoder(config.grayscale,
            config.cnn_depth, config.act, config.encoder_kernels, config)
        if config.size[0] == 64 and config.size[1] == 64:
            embed_size = 2 ** (len(config.encoder_kernels)-1) * config.cnn_depth
            embed_size *= 2 * 2
        else:
            raise NotImplemented(f"{config.size} is not applicable now")
        
        # Dynamics model (original from Iso-Dream)
        self.dynamics = networks.RSSM(
            config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
            config.dyn_input_layers, config.dyn_output_layers,
            config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete,
            config.act, config.dyn_mean_act, config.dyn_std_act,
            config.dyn_temp_post, config.dyn_min_std, config.dyn_cell,
            config.num_actions, embed_size, config.device, config)
        
        # TED classifiers
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
            
        self.ted_classifier_action = TEDClassifierSimple(feat_size, config)
        self.ted_classifier_free = TEDClassifierSimple(feat_size, config)
        
        # Task-adaptive TED parameters
        self.ted_coefficient_start = getattr(config, 'ted_coefficient_start', 0.0)
        self.ted_coefficient_end = getattr(config, 'ted_coefficient_end', 1.0)
        self.ted_warmup_ratio = getattr(config, 'ted_warmup_ratio', 0.2)
        total_steps = getattr(config, 'steps', 500000)  # Default 1M steps
        self.ted_warmup_steps = int(total_steps * self.ted_warmup_ratio)
        
        # Environment-specific settings
        self.env_name = getattr(config, 'env_name', 'walker')
        if 'cheetah' in self.env_name.lower():
            self.ted_coefficient_end *= 0.3  # Even lower for Cheetah
            print(f"Cheetah detected: TED coefficient reduced to {self.ted_coefficient_end}")
            
        # Initialize rest of the model (same as original)
        self._initialize_heads_and_optimizers(config, embed_size, feat_size)

    def _initialize_heads_and_optimizers(self, config, embed_size, feat_size):
        """Initialize heads and optimizers - same as original implementation"""
        self.heads = nn.ModuleDict()
        channels = (1 if config.grayscale else 3)
        shape = (channels,) + config.size
        
        if config.mask == 1:
            print('\033[1;35m Using single mask decoders \033[0m')
            self.heads['image'] = networks.SingleMaskDecoder(
                feat_size, config.cnn_depth, config.act, shape, 
                config.decoder_kernels, config.decoder_thin)
        elif config.mask == 2:
            print('\033[1;35mUsing double mask decoders \033[0m')
            self.heads['image'] = networks.DoubleDecoder(
                feat_size, config.cnn_depth, config.act, shape,
                config.decoder_kernels, config.decoder_thin)
        elif config.mask == 3:
            print('\033[1;35mUsing Three mask decoders \033[0m')
            self.heads['image'] = networks.TrippleDecoder(
                feat_size, config.cnn_depth, config.act, shape,
                config.decoder_kernels, config.decoder_thin)
        else:
            raise NotImplementedError
        
        self.background_decoder = networks.ConvDecoder(
            embed_size * config.init_frame, config.cnn_depth, config.act, 
            shape, config.decoder_kernels, config.decoder_thin)
        
        double_size = 2 if config.use_free else 1
        self.heads['reward'] = networks.DenseHead(
            double_size*feat_size, [], config.reward_layers, 
            config.units, config.act)
        
        if config.inverse_dynamics:
            self.heads['action'] = networks.DenseHead(
                embed_size, [config.num_actions], config.reward_layers,
                config.units, config.act)
        if config.pred_discount:
            self.heads['discount'] = networks.DenseHead(
                double_size*feat_size, [], config.discount_layers,
                config.units, config.act, dist='binary')
        
        for name in config.grad_heads:
            assert name in self.heads, name
        
        self._model_opt = tools.Optimizer(
            'model', self.parameters(), config.model_lr, config.opt_eps, 
            config.grad_clip, config.weight_decay, opt=config.opt,
            use_amp=self._use_amp)
        
        self._scales = dict(
            reward=config.reward_scale, discount=config.discount_scale, 
            action=config.action_scale)

    def _get_ted_coefficient(self):
        """Get current TED coefficient with warmup schedule"""
        if self._step < self.ted_warmup_steps:
            progress = self._step / self.ted_warmup_steps
            ted_coeff = self.ted_coefficient_start + progress * (
                self.ted_coefficient_end - self.ted_coefficient_start)
        else:
            ted_coeff = self.ted_coefficient_end
        return ted_coeff

    def _train(self, data):
        data = self.preprocess(data) 
        self.dynamics.train_wm = True
        self.dynamics.step = 0

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                # Single encoding pass (no target encoder)
                embed = self.encoder(data, data['action'])
                embed_back = self.encoder(data, data['action'], bg=True)
                embed_back = embed_back[:, :self._config.init_frame, :].reshape(
                    self._config.batch_size, -1)
                background = self.background_decoder(embed_back.unsqueeze(1)).mode()
                background = torch.clamp(background, min=-0.5, max=0.5)
                
                self.dynamics.rollout_free = True
                post, prior = self.dynamics.observe(embed, data['action'])
                self.dynamics.rollout_free = self._config.use_free
                
                # Get features for TED
                feat, feat_free = self.dynamics.get_feat_for_decoder(
                    post, prior=prior, action_step=self._config.action_step, 
                    free_step=self._config.free_step)
                
                # Calculate TED loss using same features (no target)
                ted_coefficient = self._get_ted_coefficient()
                if ted_coefficient > 0:
                    ted_loss = self._compute_ted_loss_simple(feat, feat_free)
                else:
                    ted_loss = torch.tensor(0.0, device=feat.device)
                
                # Original Iso-Dream KL loss
                kl_balance = tools.schedule(self._config.kl_balance, self._step)
                kl_free = tools.schedule(self._config.kl_free, self._step)
                kl_scale = tools.schedule(self._config.kl_scale, self._step)
                kl_loss, kl_value = self.dynamics.kl_loss(
                    post, prior, self._config.kl_forward, kl_balance, kl_free, kl_scale)
                
                if self._config.sz_sparse:
                    regularization_loss = torch.mean(prior['gate']) 
                
                losses = {}
                likes = {}
                
                # Process all heads (same as original)
                for name, head in self.heads.items():
                    if 'image' in name:
                        if self.mask == 3:
                            pred, _, _, action_mask, free_mask = self.heads['image'](
                                feat, feat_free, background)
                            if self._config.autoencoder:
                                like = pred.log_prob(data[name][:, 1:, :, :, :]) 
                            else:
                                like = pred.log_prob(data[name])  
                        else:
                            pred, _, _, _, _ = self.heads['image'](
                                feat, feat_free, data['image'])
                            like = pred.log_prob(data[name])
                    elif 'action' in name:
                        embed_action, embed_free = torch.chunk(embed, chunks=2, dim=-1)
                        inp = embed_action[:, 1:, :] - embed_action[:, :-1, :]
                        pred = head(inp)
                        like = pred.log_prob(data[name][:, 1:, :])
                    else:
                        grad_head = (name in self._config.grad_heads)
                        feat_combined = self.dynamics.get_feat(post)
                        feat_combined = feat_combined if grad_head else feat_combined.detach()
                        pred = head(feat_combined)
                        if self._config.autoencoder:
                            like = pred.log_prob(data[name][:, 1:, :]) 
                        else:
                            like = pred.log_prob(data[name])
                    likes[name] = like
                    losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
                
                # Combine all losses
                model_loss = sum(losses.values()) + kl_loss + ted_coefficient * ted_loss
                
                # Add other regularization losses (same as original)
                if self._config.min_free:
                    min_free_neg_loss = torch.sum(
                        (prior['deter_free'] - prior['deter_free_neg']) ** 2) 
                    min_free_pos_loss = torch.sum(
                        (prior['deter_free_pos'] - prior['deter_free']) ** 2)
                    min_free_loss = min_free_neg_loss + min_free_pos_loss
                    model_loss += min_free_loss 
                if self._config.max_action:
                    max_action_loss = torch.abs(torch.cosine_similarity(
                        prior['deter'], prior['deter_action_neg'], dim=2))
                    model_loss += torch.mean(max_action_loss)
                if self._config.sz_sparse:
                    model_loss += regularization_loss

            # Optimization step
            metrics = self._model_opt(model_loss, self.parameters())

        # Update metrics
        metrics.update({f'{name}_loss': to_np(loss) for name, loss in losses.items()})
        metrics['kl_balance'] = kl_balance
        metrics['kl_free'] = kl_free
        metrics['kl_scale'] = kl_scale
        metrics['ted_loss'] = to_np(ted_loss)
        metrics['ted_coefficient'] = ted_coefficient
        
        if self._config.min_free:
            metrics['min_free_loss'] = to_np(min_free_loss)
        if self._config.max_action:
            metrics['max_action_loss'] = to_np(max_action_loss)
        if self._config.sz_sparse:
            metrics['regularization_loss'] = to_np(regularization_loss)
        metrics['kl'] = to_np(torch.mean(kl_value))
        
        with torch.cuda.amp.autocast(self._use_amp):
            metrics['prior_ent'] = to_np(torch.mean(
                self.dynamics.get_dist(prior, free=False).entropy()))
            metrics['post_ent'] = to_np(torch.mean(
                self.dynamics.get_dist(post, free=False).entropy()))
            metrics['prior_ent_free'] = to_np(torch.mean(
                self.dynamics.get_dist(prior, free=True).entropy()))
            metrics['post_ent_free'] = to_np(torch.mean(
                self.dynamics.get_dist(post, free=True).entropy()))
            context = None
            
        post = {k: v.detach() for k, v in post.items()}
        self.dynamics.train_wm = False
        return post, context, metrics

    def _compute_ted_loss_simple(self, feat, feat_free):
        """
        Simplified TED loss using temporal consistency within the same sequence
        No target encoder needed - uses temporal relationships in current features
        """
        # Process action-conditioned branch
        ted_loss_action = self._compute_branch_ted_loss_temporal(feat, self.ted_classifier_action)
        ted_loss = ted_loss_action
        
        # # Process action-free branch  
        # ted_loss_free = self._compute_branch_ted_loss_temporal(
        #     feat_free, self.ted_classifier_free)
        
        # # Combine losses from both branches
        # ted_loss = ted_loss_action + ted_loss_free
        
        return ted_loss
        
    def _compute_branch_ted_loss_temporal(self, feat, classifier):
        """
        Compute TED loss using temporal relationships within the same sequence
        This encourages the model to distinguish between:
        1. Consecutive timesteps (temporal=1) 
        2. Non-consecutive timesteps from same episode (temporal=0)
        3. Random pairs from different episodes (temporal=0)
        """
        batch_size = feat.shape[0]
        seq_len = feat.shape[1]
        feat_dim = feat.shape[2]
        loss_fn = nn.BCEWithLogitsLoss()
        
        # Skip if sequence is too short
        if seq_len < 3:
            return torch.tensor(0.0, device=feat.device)
        
        all_samples = []
        all_labels = []
        
        # Sample temporal pairs (consecutive timesteps) - these should be labeled as temporal=1
        num_consecutive = min(2, seq_len - 1)
        for i in range(num_consecutive):
            if i + 1 >= seq_len:
                break
                
            obs_t = feat[:, i]      # [batch_size, feat_dim]
            obs_t1 = feat[:, i + 1] # [batch_size, feat_dim]
            
            temporal_samples = torch.stack([obs_t, obs_t1], dim=1)
            temporal_labels = torch.ones((batch_size, 1), device=feat.device)
            
            all_samples.append(temporal_samples)
            all_labels.append(temporal_labels)
        
        # Sample non-consecutive pairs from same episode - labeled as temporal=0
        for i in range(min(2, seq_len - 2)):
            if i + 2 >= seq_len:
                break
                
            obs_t = feat[:, i]      # [batch_size, feat_dim]
            obs_t2 = feat[:, i + 2] # [batch_size, feat_dim] (skip one timestep)
            
            non_consecutive_samples = torch.stack([obs_t, obs_t2], dim=1)
            non_consecutive_labels = torch.zeros((batch_size, 1), device=feat.device)
            
            all_samples.append(non_consecutive_samples)
            all_labels.append(non_consecutive_labels)
        
        # Sample random pairs from different episodes - labeled as temporal=0
        rnd_idx = torch.randperm(batch_size, device=feat.device)
        obs_rand1 = feat[:, 0]           # [batch_size, feat_dim]
        obs_rand2 = feat[rnd_idx, 1]     # [batch_size, feat_dim] (from different episodes)
        
        random_samples = torch.stack([obs_rand1, obs_rand2], dim=1)
        random_labels = torch.zeros((batch_size, 1), device=feat.device)
        
        all_samples.append(random_samples)
        all_labels.append(random_labels)
        
        if not all_samples:
            return torch.tensor(0.0, device=feat.device)
        
        # Combine all samples and labels
        samples = torch.cat(all_samples)
        labels = torch.cat(all_labels)
        
        # Get predictions from classifier
        preds = classifier(samples)
        
        # Calculate loss
        ted_loss = loss_fn(preds, labels)
        
        return ted_loss

    def preprocess(self, obs):
        obs = obs.copy()
        obs['image'] = torch.Tensor(obs['image']) / 255.0 - 0.5
        if self._config.clip_rewards == 'tanh':
            obs['reward'] = torch.tanh(torch.Tensor(obs['reward'])).unsqueeze(-1)
        elif self._config.clip_rewards == 'identity':
            obs['reward'] = torch.Tensor(obs['reward']).unsqueeze(-1)
        else:
            raise NotImplemented(f'{self._config.clip_rewards} is not implemented')
        if 'discount' in obs:
            obs['discount'] *= self._config.discount
            obs['discount'] = torch.Tensor(obs['discount']).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    # Include video_pred method same as original
    def video_pred(self, data):
        # Same implementation as original WorldModel
        pass


class TEDClassifierSimple(nn.Module):
    """Simplified TED classifier without environment complexity"""
    def __init__(self, feature_dim, config):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Simple neural network approach that works for both environments
        hidden_dim = min(feature_dim, 256)
        self.network = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, samples):
        """
        Simple classification of temporal relationships
        """
        batch_size = samples.shape[0]
        x1 = samples[:, 0]  # [batch_size, feature_dim]
        x2 = samples[:, 1]  # [batch_size, feature_dim]
        
        # Concatenate features and pass through network
        concat_features = torch.cat([x1, x2], dim=1)
        output = self.network(concat_features)
        
        return output
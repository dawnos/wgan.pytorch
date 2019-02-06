
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

class WGANTrainer(object):
  def __init__(self, net, args):
    self.net = net
    self.gen_optimizer = optim.Adam(net.gen.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    self.dis_optimizer = optim.Adam(net.dis.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

  def train(self, data_loader, args):

    it = iter(data_loader)
    for step in range(args.iters):

      real_data = next(it)
      fake_data = self.net.gen()

      pred_real = self.net.dis(real_data)
      pred_fake = self.net.dis(fake_data)
      
      # Loss
      dis_loss = torch.mean(pred_fake) - torch.mean(pred_real)
      gen_loss = -torch.mean(pred_fake)

      # Update discriminator
      alpha = torch.rand([args.batch_size, 1])
      interpolates = alpha * real_data + (1-alpha) * fake_data
      pred_interpolates = self.net.dis(interpolates)
      gradients = autograd.grad(pred_interpolates, interpolates, 
        grad_outputs=torch.ones(interpolates.size()), 
        retain_graph=True, create_graph=True, only_inputs=True)[0]
      # slopes = torch.sqrt(torch.sum(torch.pow(gradients, 2), dim=1))
      slopes = torch.norm(gradients, dim=1)
      gradient_penalty = torch.mean((slopes - 1)**2)
      dis_loss += args.lambda1 * gradient_penalty
      self.dis_optimizer.zero_grad()
      dis_loss.backward(retain_graph=True)
      self.dis_optimizer.step()

      # Update generator
      if step % args.critic_iters == 0:
        self.gen_optimizer.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()

      if step % 100 == 0:
        plt.clf()
        plt.plot(real_data[0,:], real_data[1,:])

      print("dis_loss=%f, gen_loss=%f" % (dis_loss.item(), gen_loss.item()))

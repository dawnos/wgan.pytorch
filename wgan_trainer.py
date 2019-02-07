import numpy as np
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import torch
import torch.optim as optim
import torch.autograd as autograd


class WGANTrainer(object):
  def __init__(self, net, args):
    self.net = net
    if args.optimizer == "Adam":
      self.gen_optimizer = optim.Adam(net.gen.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
      self.dis_optimizer = optim.Adam(net.dis.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer == "":
      self.gen_optimizer = optim.Adam(net.gen.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
      self.dis_optimizer = optim.Adam(net.dis.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

  def train(self, data_loader, args):

    it = iter(data_loader)
    for step in range(args.iters):

      real_data = next(it).to(args.device)
      fake_data = self.net.gen()

      pred_real = self.net.dis(real_data)
      pred_fake = self.net.dis(fake_data)

      # Loss
      dis_loss = torch.mean(pred_fake) - torch.mean(pred_real)
      gen_loss = -torch.mean(pred_fake)

      # Update discriminator
      alpha = torch.rand([args.batch_size, 1]).to(args.device)
      interpolates = alpha * real_data + (1 - alpha) * fake_data
      pred_interpolates = self.net.dis(interpolates)
      gradients = autograd.grad(pred_interpolates, interpolates,
                                grad_outputs=torch.ones(interpolates.size()).to(args.device),
                                retain_graph=True, create_graph=True, only_inputs=True)[0]
      # slopes = torch.sqrt(torch.sum(torch.pow(gradients, 2), dim=1))
      slopes = torch.norm(gradients, dim=1)
      gradient_penalty = torch.mean((slopes - 1) ** 2)
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

        self.net.eval()

        with torch.no_grad():
          points = np.zeros((args.map_point_num, args.map_point_num, 2), dtype='float32')
          points[:, :, 0] = np.linspace(-args.map_point_range, args.map_point_range, args.map_point_num)[:, None]
          points[:, :, 1] = np.linspace(-args.map_point_range, args.map_point_range, args.map_point_num)[None, :]
          points = points.reshape((-1, 2))
          pred_points = self.net.dis(torch.FloatTensor(points).to(args.device)).cpu().numpy()

          plt.clf()
          x = y = np.linspace(-args.map_point_range, args.map_point_range, args.map_point_num)
          plt.contour(x, y, pred_points.reshape((len(x), len(y))).transpose())
          real_data = real_data.cpu().detach().numpy()
          plt.plot(real_data[:, 0], real_data[:, 1], '+', color='orange')
          fake_data = fake_data.cpu().detach().numpy()
          plt.plot(fake_data[:, 0], fake_data[:, 1], '+', color='green')
          plt.title('Step=%d' % step)
          # plt.show()
          plt.savefig("log/map_%06d.png" % step, format="png")

        self.net.train()

      print("step=%d/%d, dis_loss=%f, gen_loss=%f" % (step, args.iters, dis_loss.item(), gen_loss.item()))

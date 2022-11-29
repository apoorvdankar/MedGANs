from torch import nn
from torchvision import transforms 
from torchvision.transforms import InterpolationMode


# Defining the Generator Block
def get_gen_blockConv(in_channels, out_channels, kernel_size, stride, padding=0):
  model = nn.Sequential(
      # Convolution Layer
      nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
      # Batch Normalization
      nn.BatchNorm2d(out_channels),
      # PReLU Activation function
      nn.PReLU()
  )
  return model

class Generator(nn.Module):

  def __init__(self):
    super(Generator, self).__init__()

    self.block1 = get_gen_blockConv(1, 16, (9,9), 1, padding='same')
    self.block2 = get_gen_blockConv(16, 32, (3,3), 1, padding='same')
    self.block3 = get_gen_blockConv(32, 64, (3,3), 1, padding='same')
    self.block4 = get_gen_blockConv(64, 128, (3,3), 1, padding='same')
    self.blockResConv = get_gen_blockConv(128, 128, (3,3), 1, padding='same')
    self.block5= get_gen_blockConv(128, 64, (3,3), 1, padding='same' )
    self.block6 = get_gen_blockConv(64, 32, (3,3), 1, padding='same' )
    self.block7 = get_gen_blockConv(32, 16, (3,3), 1, padding='same' )
    self.lastconv = get_gen_blockConv(16, 1, (9,9), 1, padding='same' )
    self.tanh = nn.Tanh()
    self.prelu = nn.PReLU()

  def resnet_forward(self, input):
    output = self.blockResConv(input)

    output = output + input
    output = self.prelu(output)
    return output

  def forward(self,noisy_image):
    # Down sampling
    conv1 = self.block1(noisy_image)
    conv2 = self.block2(conv1)
    conv3 = self.block3(conv2)
    conv4 = self.block4(conv3)

    # ResNet Function
    res1 = self.resnet_forward(conv4)
    res2 = self.resnet_forward(res1)
    res3 = self.resnet_forward(res2)

    # Up sampling
    T0 = transforms.Resize((32, 32), InterpolationMode.NEAREST)
    deconv0 = self.block5(T0(res3))
    T1 = transforms.Resize((64, 64), InterpolationMode.NEAREST)
    deconv1 = self.block6(T1(deconv0))
    T2 = transforms.Resize((128, 128), InterpolationMode.NEAREST)
    deconv2 = self.block7(T2(deconv1))
    deconv2 = deconv2 + conv1
    conv5 = self.lastconv(deconv2)
    conv5= self.tanh(conv5)
    output = conv5 + noisy_image

    return output
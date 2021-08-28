import torch,os
import torch.nn as nn
from torch.nn.parameter import Parameter

class Multi_Context(nn.Module):
    def __init__(self, inchannels):
        super(Multi_Context, self).__init__()
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True))
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels * 3, out_channels=inchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannels))

    def forward(self, x):
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x = torch.cat([x1,x2,x3], dim=1)
        x = self.conv2(x)
        return x

class Adaptive_Weight(nn.Module):
    def __init__(self, inchannels):
        super(Adaptive_Weight, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.inchannels = inchannels
        self.fc1 = nn.Conv2d(inchannels, inchannels//4, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(inchannels//4, 1, kernel_size=1, bias=False)
        self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = self.avg(x)
        weight = self.relu1(self.fc1(x_avg))
        weight = self.relu2(self.fc2(weight))
        weight = self.sigmoid(weight)
        out = x * weight
        return out

class Counter_attention(nn.Module):
    def __init__(self, inchannels):
        super(Counter_attention, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(inchannels))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(inchannels))
        # self.conv3 = nn.Sequential(nn.Conv2d(in_channels=inchannels*2, out_channels=inchannels, kernel_size=1),
        #                            nn.BatchNorm2d(inchannels))
        self.sig = nn.Sigmoid()
        self.mc1 = Multi_Context(inchannels)
        self.mc2 = Multi_Context(inchannels)
        self.ada_w1 = Adaptive_Weight(inchannels)
        self.ada_w2 = Adaptive_Weight(inchannels)
    def forward(self, assistant, present):

        mc1 = self.mc1(assistant)
        pr1 = present * self.sig(mc1)
        pr2 = self.conv1(present)
        pr2 = present * self.sig(pr2)
        out1 = pr1 + pr2 + present


        mc2 = self.mc2(present)
        as1 = assistant * self.sig(mc2)
        as2 = self.conv2(assistant)
        as2 = assistant * self.sig(as2)
        out2 = as1 + as2 + assistant


        out1 = self.ada_w1(out1)
        out2 = self.ada_w2(out2)
        out = out1 + out2

        # out = torch.cat([out1, out2], dim=1)
        # out = self.conv3(out)

        return out

class Counter_Guide(nn.Module):
    def __init__(self):
        super(Counter_Guide, self).__init__()
        self.counter_atten1 = Counter_attention(128)
        self.counter_atten2 = Counter_attention(256)



    def forward(self, frame1, frame2, event1, event2):
        out1 = self.counter_atten1(frame1, event1)
        out2 = self.counter_atten2(frame2, event2)

        return out1, out2


if __name__ == '__main__':
    net = Counter_Guide()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = net.cuda()

    var1 = torch.FloatTensor(10, 128, 36, 36).cuda()
    var2 = torch.FloatTensor(10, 256, 18, 18).cuda()
    var3 = torch.FloatTensor(10, 128, 36, 36).cuda()
    var4 = torch.FloatTensor(10, 256, 18, 18).cuda()
    # var = Variable(var)

    out1, out2 = net(var1, var2, var3, var4)

    print('*************')
    print(out1.shape, out2.shape)

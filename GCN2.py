import tensorflow as tf
import torch

# new_data = torch.IntTensor(a[0])

# print(new_data,b,a)
import torch
import math
import numpy as np
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN_thnh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(9, 4)
        self.conv2 = GCNConv(4, 2)
    #         self.conv3 = GCNConv(4,2)
    # self.classifier = Linear(2,dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h=h.tanh()
        h=self.conv2(h,edge_index)
        #         h=h.tanh()
        #         h=self.conv3(h,edge_index)
        # h=h.tanh()

        #         #分类层
        #         out = self.classifier(h)

        return h


# 智能体的操作流程，循环学习多少次；先初始化环境，得到初始状态，agent得到动作；env执行动作得到新的状态奖励等
class GCN_sigm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(9, 4)

        self.conv2 = GCNConv(4, 2)
    #         self.conv3 = GCNConv(4,2)
    # self.classifier = Linear(2,dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h=h.sigmoid()
        h=self.conv2(h,edge_index)
        #         h=h.tanh()
        #         h=self.conv3(h,edge_index)
        # h=h.tanh()

        #         #分类层
        #         out = self.classifier(h)

        return h




class GCN_3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(9, 4)

        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4,2)
    # self.classifier = Linear(2,dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h=h.tanh()
        h=self.conv2(h,edge_index)
        h=h.tanh()
        h=self.conv3(h,edge_index)
        # h=h.tanh()

        #         #分类层
        #         out = self.classifier(h)

        return h
class GCN_4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(9, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 4)
        self.conv4 = GCNConv(4, 2)
    #         self.conv3 = GCNConv(4,2)
    # self.classifier = Linear(2,dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h=h.tanh()
        h=self.conv2(h,edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        h = self.conv4(h, edge_index)
        #         h=h.tanh()
        #         h=self.conv3(h,edge_index)
        # h=h.tanh()

        #         #分类层
        #         out = self.classifier(h)

        return h
class GCN_5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(9, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 4)
        self.conv4 = GCNConv(4, 4)
        self.conv5 = GCNConv(4, 2)
    #         self.conv3 = GCNConv(4,2)
    # self.classifier = Linear(2,dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h=h.tanh()
        h=self.conv2(h,edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        h = self.conv4(h, edge_index)
        h = h.tanh()
        h = self.conv5(h, edge_index)
        #         h=h.tanh()
        #         h=self.conv3(h,edge_index)
        # h=h.tanh()

        #         #分类层
        #         out = self.classifier(h)

        return h
class GCN_6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(9, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 4)
        self.conv4 = GCNConv(4, 4)
        self.conv5 = GCNConv(4, 4)
        self.conv6 = GCNConv(4, 2)
    #         self.conv3 = GCNConv(4,2)
    # self.classifier = Linear(2,dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h=h.tanh()
        h=self.conv2(h,edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        h = self.conv4(h, edge_index)
        h = h.tanh()
        h = self.conv5(h, edge_index)
        h = h.tanh()
        h = self.conv6(h, edge_index)
        #         h=h.tanh()
        #         h=self.conv3(h,edge_index)
        # h=h.tanh()

        #         #分类层
        #         out = self.classifier(h)

        return h
GCN_model = GCN_thnh()
GCN_model_sig = GCN_sigm()
GCN_model_3 = GCN_3()
GCN_model_4 = GCN_4()
GCN_model_5 = GCN_5()
GCN_model_6 = GCN_6()



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


class Discriminator(nn.Module):

    def __init__(self, inchannel, outchannel, num_classes, patch_size,  LABEL_VALUES_src ):
        super(Discriminator, self).__init__()
        dim = 512
        batch_size = 64
        self.LABEL_VALUES_src = LABEL_VALUES_src
        self.patch_size = patch_size
        self.inchannel = inchannel

        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv1_fft = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.relu1_fft = nn.ReLU(inplace=True)
        self.conv2_fft = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2_fft = nn.ReLU(inplace=True)
        self.fc1_fft = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3_fft = nn.ReLU(inplace=True)
        self.fc2_fft = nn.Linear(dim, dim)
        self.relu4_fft = nn.ReLU(inplace=True)

        self.cls_head_src = nn.Linear(int(dim), int(num_classes))
        self.sel_fft_num = nn.Linear(256, 1)
        self.p_mu = nn.Linear(dim, outchannel, nn.LeakyReLU())
        self.pro_head = nn.Linear(dim, outchannel, nn.ReLU())
        self.sel_FFT = nn.Linear(int(dim), int(num_classes))

        # 3D
        self.conv1_3d = nn.Conv3d(1, 20, (3, 3, 3), stride=(1, 1, 1), padding=0)
        self.pool1_3d = nn.Conv3d(20, 20, (3, 1, 1), stride=(2, 1, 1), padding=0)
        self.conv2_3d = nn.Conv3d(20, 35, (3, 3, 3), stride=(1, 1, 1), padding=0)
        self.pool2_3d = nn.Conv3d(35, 35, (3, 1, 1), stride=(2, 1, 1), padding=0)
        # 1D
        self.conv3_1d = nn.Conv3d(35, 35, (3, 1, 1), stride=(1, 1, 1), padding=0)
        self.conv4_1d = nn.Conv3d(35, 35, (2, 1, 1), stride=(2, 1, 1), padding=0)
        self.conv5_cat = nn.Conv2d(inchannel * 2, inchannel, kernel_size=1)

        self.fc1_3d = nn.Linear(self._get_final_flattened_size_3d(), dim)
        self.fc2_3d = nn.Linear(dim, outchannel)

        self.cat_dim1 = nn.Linear(dim * 2, dim)

        self.fft_sel_head = nn.Linear(dim, outchannel)
        self.el_dif = nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1)

        self.fc1_forU = nn.Linear(batch_size, int(batch_size / 2))
        self.fc2_forU = nn.Linear(int(batch_size / 2), int(batch_size / 4))
        self.fc3_forU = nn.Linear(int(batch_size / 4), 1)
        self.relu1_U = nn.ReLU(inplace=True)
        self.relu2_U = nn.ReLU(inplace=True)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def _get_final_flattened_size_3d(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.inchannel, self.patch_size, self.patch_size)
            )
            x = self.pool1_3d(self.conv1_3d(x))
            x = self.pool2_3d(self.conv2_3d(x))
            x = self.conv3_1d(x)
            x = self.conv4_1d(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def get_top_indices(self, x):
        x = x.to(device=torch.device('cuda'), dtype=torch.float32)

        x_sor = x

        transformed_data = fft.fftn(x, dim=(-3, -2, -1))
        x = torch.real(transformed_data)

        x = x + x_sor

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))
        out5 = self.fft_sel_head(out4)
        x = torch.sum(out5, dim=1)

        in_size_1 = x.size(0)
        x_u = x
        batch_size = 256
        if in_size_1 < batch_size:
            padded_x = torch.zeros(batch_size, device=x.device)
            padded_x[:x_u.size(0)] = x_u
            x_u = padded_x
        else:
            pass

        x_u1 = self.relu1_U(self.fc1_forU(x_u))
        x_u2 = self.relu2_U(self.fc2_forU(x_u1))
        x_u3 = self.fc3_forU(x_u2)
        u = torch.sigmoid(x_u3)

        max_x = x.max()
        min_x = x.min()
        threshold = min_x + u * (max_x - min_x)

        mask = (x >= threshold).to(torch.int)
        indices = mask.nonzero(as_tuple=False).squeeze()

        return indices

    def forward(self, x, mode='test'):
        x = x.to(device=torch.device('cuda'), dtype=torch.float32)

        top_indices = self.get_top_indices(x)

        selected_patches = x[top_indices]

        if selected_patches.size(0) > 0:
            selected_patches_fft = fft.fftn(selected_patches, dim=(-3, -2, -1))

            selected_patches_fft = torch.real(selected_patches_fft)

            x = x.clone()
            x[top_indices] = selected_patches_fft
        else:
            pass

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if mode == 'test':
            clss = self.cls_head_src(out4)
            return clss
        elif mode == 'train':
            proj = F.normalize(self.pro_head(out4))
            clss = self.cls_head_src(out4)

            return clss, proj


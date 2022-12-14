import torch
import numpy as np
import torch.nn as nn
import itertools


import torch
import numpy as np
import torch.nn as nn
import itertools
import data_loader as dl
from scipy import signal
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

# from torchsummary import summary

torch.cuda.empty_cache()

window_len = 2048  # 128 ms in 1 sec length of signal
time_segment_stft = 33  # For 1 second of signal and 50 % overlap
max_channel_data = 2
device = "cpu"
ACT_NONE = 0
_STAGE1_LOSS = nn.MSELoss()
batch_size = 1
no_of_epochs = 110
sigmoid = nn.Sigmoid()


def cal_features(data):

    data_stft_real = np.zeros(
        (data.shape[0], max_channel_data, window_len // 2 + 1, time_segment_stft)
    )
    data_stft_imag = np.zeros(
        (data.shape[0], max_channel_data, window_len // 2 + 1, time_segment_stft)
    )

    for batch in np.arange(data.shape[0]):
        data_stft = signal.stft(
            data[batch, :, :], axis=-1, fs=16000, nperseg=window_len
        )

        data_stft_real[batch, :, :, :] = np.real(data_stft[2])
        data_stft_imag[batch, :, :, :] = np.imag(data_stft[2])

    x = np.concatenate((data_stft_real, data_stft_imag), axis=1)
    x = np.transpose(
        x, (0, 1, 3, 2)
    )  # Dimension (Batch_size x 4 {2 real and 2 imag coeffs} x time_frames x freq_bins )

    return torch.tensor(x).to(device=device)


def stage_1_loss(multi_dimensional_data, gt_azi):

    squeeze_multi_dimensional_data = torch.squeeze(
        multi_dimensional_data
    )  # Remove the dimension of the extra pipeline

    (
        ndata,
        nfbin,
        nframe,
        ndoa,
    ) = (
        squeeze_multi_dimensional_data.size()
    )  # Dimension (batch_size x nfbin x ntimeframe x 360)
    gndata, gndoa = gt_azi.size()  # Dimension (batch_size x 360)

    egt_azi = gt_azi.expand((nframe, nfbin, ndata, ndoa)).permute(
        2, 1, 0, 3
    )  # Expand the same values in other dimensions

    loss_theta = _STAGE1_LOSS(
        squeeze_multi_dimensional_data, egt_azi
    )  # Calculate the loss

    return loss_theta


def loss_end_2_end(output, gt_theta):

    theta_loss = _STAGE1_LOSS(output, gt_theta)
    # theta_loss = ((output[:, 0] - gt_theta[:, 0]) ** 2.0).mean() # Mean square loss between dimension of (batch_size x doa)

    return theta_loss


tolerance = 10
eta = 0.5
"""
def doa_metric_unknown_sources(output,gt_azi):
    grid_az=np.linspace(0,180,num=360)
    values_greater_than_threshold=output>eta
    angles_for_values_greater_than_threshold=grid_az[values_greater_than_threshold]
    distance_within_tolerance=np.abs(angles_for_values_greater_than_threshold-gt_azi)<tolerance
    index_=np.nonzero(output>eta)[0][distance_within_tolerance]

    if index_.shape[0] == 0:
        index_=np.argmin(np.abs(angles_for_values_greater_than_threshold-gt_azi))



    predicted_value=grid_az[index_[np.argmax(output[index_])]]

    error=np.abs(predicted_value-gt_azi)

    return error
"""

# Known number of sources
def doa_metric(predicted_spectrum, azimuth_gt, switch="train"):
    batch_acc = 0
    mean_batch_err = 0
    mean_batch_loc = 0
    grid_az = np.linspace(0, 180, num=360)
    predicted_angle_arr = []
    batch_size = predicted_spectrum.shape[0]
    print(predicted_spectrum.shape)
    # predicted_spectrum=predicted_spectrum.detach().cpu().numpy()   #Only for voicehome2

    for no in np.arange(batch_size):

        index_ = np.argmax(predicted_spectrum[no, :])

        predicted_angle = grid_az[index_]
        predicted_angle = np.abs(
            180 - predicted_angle
        )  # For DIRHA use this, for Dcase change this

        if "val" in switch or "test" in switch:
            predicted_angle_arr.append([predicted_angle, azimuth_gt[no]])

        if np.abs(predicted_angle - azimuth_gt[no][0]) <= tolerance:
            batch_acc = batch_acc + 1
            mean_batch_err = (
                np.abs(predicted_angle - azimuth_gt[no][0]) + mean_batch_err
            )
            mean_batch_loc = (
                np.abs(predicted_angle - azimuth_gt[no][0]) + mean_batch_loc
            )
        else:
            mean_batch_err = (
                np.abs(predicted_angle - azimuth_gt[no][0]) + mean_batch_err
            )

    if "val" in switch or "test" in switch:
        return (
            ((batch_acc * 100) / batch_size),
            (mean_batch_err / batch_size),
            predicted_angle_arr,
            (mean_batch_loc / batch_acc),
        )
    else:
        return ((batch_acc * 100) / batch_size), (mean_batch_err / batch_size)


# Residual block....
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        seq = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        self.mseq = nn.Sequential(*seq)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.mseq(x)

        residual = x
        out += residual
        out = self.relu(out)
        return out


class he_archs(nn.Module):
    def __init__(
        self,
        input_size,
        output_act=ACT_NONE,
        n_out_map=1,
        s2_hidden_size=[500],
        s2_azi_nsize=5,
        output_size=360,
        n_res_blocks=5,
        roll_padding=True,
    ):

        super().__init__()

        self.output_size = output_size  # 360 the size of the spectrum
        self.s2_azi_nsize = s2_azi_nsize  #
        self.roll_padding = roll_padding
        self.feat_layers = 1
        self.drp_1 = nn.Dropout(p=0.4)

        ic, x, y = input_size  # Channel, nframe, nfbin

        # stage one:
        s1seq = []

        # initial layers (no residual)
        # layer 1
        oc = 4 * ic  # Four times the number of input channels

        # s1seq.append(nn.LayerNorm([1025]))
        s1seq.append(
            nn.Conv2d(ic, oc, kernel_size=(1, 7), stride=(1, 3))
        )  # ic:4 , oc:16
        # s1seq.append(nn.LayerNorm([340]))
        s1seq.append(nn.ReLU(inplace=True))
        s1seq.append(nn.Dropout(p=0.4))

        ic = oc  # ic becomes 16
        x = x  # nframe
        y = (y - 7 + 3) // 3  # nfbin changes coz of convolution layer

        # layer 2
        oc = 4 * ic  # Four times the number of output channel of layer 1

        # oc becomes 16*4 = 64

        s1seq.append(
            nn.Conv2d(ic, oc, kernel_size=(1, 5), stride=(1, 2))
        )  # ic : 16, oc : 64
        # s1seq.append(nn.LayerNorm([168]))
        s1seq.append(nn.ReLU(inplace=True))
        s1seq.append(nn.Dropout(p=0.4))

        ic = oc  # ic = 64
        x = x  # nframe
        y = (y - 5 + 2) // 2  # nfbin changes coz of convolution layer

        # residual layers ic = oc , 5 residual layers
        for _ in range(n_res_blocks):
            s1seq.append(ResidualBlock(ic, oc))

        # stage one trunk
        self.stage1trunk = nn.Sequential(*s1seq)

        # stage one output: map to 360 directions
        # output size should depend on roll_padding

        if roll_padding:
            s1_output_size = output_size
        else:
            s1_output_size = output_size + s2_azi_nsize - 1

        # ic = 64, s1_output_size = 360
        # Output of stage1trunk goes to stage1 out, depending on the number of output we want, this layer implements 1 or 2 convolution layers.

        self.stage1out = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ic, s1_output_size, kernel_size=1),
                    # nn.LayerNorm([168]),
                    nn.ReLU(inplace=True),
                )
                for _ in range(n_out_map)
            ]
        )

        # stage 2
        s2seqs = []
        for _ in range(n_out_map):
            s2seq = []

            # input size: nfbin, nframe, ndoa
            ic = y  # Input size does not changes from output of stage 1 as here we refer ic as no of frequency bins, in the previous stage ic is used for number of channels in the input

            # hidden layers
            # ic = nfbin
            # oc = 500
            for oc in s2_hidden_size:
                s2seq.append(nn.Conv2d(ic, oc, kernel_size=1))
                # s2seq.append(nn.LayerNorm([364]))
                s2seq.append(nn.ReLU(inplace=True))
                s2seq.append(nn.Dropout(p=0.4))
                ic = oc

            # output layer
            # output size: 1, 1, 360 ic = 500
            oc = 1
            # print("Kernel size", x, s2_azi_nsize)
            s2seq.append(nn.Conv2d(ic, oc, kernel_size=(x, s2_azi_nsize)))

            # An activation layer can be added
            # s2seq.append()

            s2seqs.append(s2seq)
        self.stage2 = nn.ModuleList([nn.Sequential(*s) for s in s2seqs])

    # Two more functions for stage 1

    def _stage1_internal(self, x):
        # input  : ndata, nch, nframe, nfbin

        x = self.stage1trunk(x)

        # Use this line when you want to have multiple outputs same like the paper.

        x = torch.stack([branch(x) for branch in self.stage1out])

        x = self.drp_1(x)

        # print(x.shape)
        # x = self.stage1out[0](x)
        # print("Before permute stage 1",x.shape)
        # now    : n_out_map, ndata, ndoa, nframe, nfbin
        # output : ndata, nfbin, nframe, n_out_map, ndoa
        # return x.permute(0,3,2,1) #output : ndata, nfbin, nframe, ndoa
        return x.permute(1, 4, 3, 0, 2)

    # This function is called during first stage of training process
    def stage1(self, x):
        s1_internal = self._stage1_internal(x)
        if self.roll_padding:
            return s1_internal
        else:
            # stage 1 output consider padding when roll_padding=False
            return s1_internal.narrow(4, self.s2_azi_nsize // 2, self.output_size)

    # This function is called when we train end-to-end .
    def forward(self, x):
        # input  : ndata, nch, nframe, nfbin

        ######################################
        # Normalizing input data, V.Imp Step  #
        ######################################

        for bb in np.arange(batch_size):
            x[bb, :, :, :] = (x[bb, :, :, :] - torch.mean(x[bb, :, :, :])) / torch.std(
                x[bb, :, :, :]
            )

        x = self._stage1_internal(x)
        # print("After stage 1",x.shape)
        # now    : ndata, nfbin, nframe, n_out_hidden, ndoa
        # padding

        if self.roll_padding:
            x = torch.cat((x, x.narrow(4, 0, self.s2_azi_nsize - 1)), dim=4)

        # print("Before Stage 2",x.shape)
        # stage two:
        output = torch.stack(
            [branch(x[:, :, :, bid, :]) for bid, branch in enumerate(self.stage2)],
            dim=3,
        )
        # print("Output",output.shape)
        assert output.size(1) == 1
        assert output.size(2) == 1
        assert output.size(3) == len(self.stage2)
        # output : ndoa, n_out_hidden, ndoa
        if output.size(3) > 1:
            return sigmoid(output[:, 0, 0])
        else:
            return sigmoid(output[:, 0, 0, 0])

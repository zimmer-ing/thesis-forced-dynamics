import numpy as np

from typing import Dict, Optional
import Constants as const
from src.datasets.dataset_base import MultiFileTimeSeriesDataset
from pathlib import Path
import pandas as pd
from scipy import signal


from plotly import express as px

import torch
from torchdiffeq import odeint


class Order_4_RLC_Circuit(torch.nn.Module):
    def __init__(self, learn_params=False,
                 C1=500 * 1e-6,  #500 micro farad
                 C2=100 * 1e-6,  #100 micro farad
                 R1=50 * 1e-3,  #50 milli ohm
                 R2=10 * 1e-6,  #10 micro ohm
                 Rp1=1 * 1e9,  #1 giga ohm
                 Rp2=1 * 1e9,  #1 giga ohm
                 L1=1 * 1e-6,  #1 micro henry
                 L2=100 * 1e-9,  #100 nano henry
                 Nonlinear_C1=False,
                 Nonlinear_C2=True,
                 Nonlinear_L1=True,
                 Nonlinear_L2=False,
                 Slope_C1=0.1,
                 Slope_C2=0.1,
                 Slope_L1=3,
                 Slope_L2=0.1):
        super(Order_4_RLC_Circuit, self).__init__()
        self.nonlinear_C1 = Nonlinear_C1
        self.nonlinear_C2 = Nonlinear_C2
        self.nonlinear_L1 = Nonlinear_L1
        self.nonlinear_L2 = Nonlinear_L2




        self.C1 = torch.tensor(C1)
        self.C2 = torch.tensor(C2)
        self.R1 = torch.tensor(R1)
        self.R2 = torch.tensor(R2)
        self.L1 = torch.tensor(L1)
        self.L2 = torch.tensor(L2)
        self.slope_c1 = torch.tensor(Slope_C1)
        self.slope_c2 = torch.tensor(Slope_C2)
        self.slope_l1 = torch.tensor(Slope_L1)
        self.slope_l2 = torch.tensor(Slope_L2)


    def get_parameters(self):

        return {'C1': self.C1.item(),
                'C2': self.C2.item(),
                'R1': self.R1.item(),
                'R2': self.R2.item(),
                'L1': self.L1.item(),
                'L2': self.L2.item(),
                'slope_c1': self.slope_c1.item(),
                'slope_c2': self.slope_c2.item(),
                'slope_l1': self.slope_l1.item(),
                'slope_l2': self.slope_l2.item()}

    def print_params(self):

        print(f"C1={self.C1},C2={self.C2},R1={self.R1},R2={self.R2},L1={self.L1},L2={self.L2},slope_c1={self.slope_c1},slope_c2={self.slope_c2},slope_l1={self.slope_l1},slope_l2={self.slope_l2}")

    def nonlinear_capacitor(self, voltage, C, slope=0.1):
        c_act = torch.where(voltage >= 0,
                            C * torch.exp(-slope * voltage),
                            C * torch.exp(slope * voltage))
        return c_act

    def nonlinear_inductor(self, current, L, slope_l=3):
        # Nonlinear inductance effect, if current is near 0, l_act is L
        if current.abs() < 1e-6:
            l_act = torch.Tensor([L])
        else:
            tanh_term = torch.tanh(slope_l * current)
            l_act = L * tanh_term / (slope_l * current)
        return l_act


    def set_x(self, t, x):
        self.t = t
        self.x = x

    def get_current_x(self, t_query):
        return self._batch_linear_interpolation(t_query)

    def _batch_linear_interpolation(self, t_query):

        idx_left = (torch.abs(self.t - t_query)).argmin().item()
        if idx_left + 1 >= self.t.shape[0]:
            return self.x[:, idx_left, :]
        idx_right = idx_left + 1

        x_left = self.x[:, idx_left, :]
        x_right = self.x[:, idx_right, :]

        t_left = self.t[idx_left]
        t_right = self.t[idx_right]

        interpolated_x = x_left + (x_right - x_left) * ((t_query - t_left) / (t_right - t_left))

        return interpolated_x.squeeze(1)

    def forward(self, t, z, x=None):
        if x is None:
            x = self.get_current_x(t)

        v_IN = x[..., 0]
        i_OUT = x[..., 1]
        v_OUT = z[..., 0]
        i_IN = z[..., 1]
        v_INT = z[..., 2]
        i_INT = z[..., 3]

        C1 = self.C1
        C2 = self.C2
        R1 = self.R1
        R2 = self.R2
        L1 = self.L1
        L2 = self.L2
        slope_c1 = self.slope_c1
        slope_c2 = self.slope_c2
        slope_l1 = self.slope_l1
        slope_l2 = self.slope_l2

        C1_eff = self.nonlinear_capacitor(v_INT, C1, slope_c1) if self.nonlinear_C1 else C1
        C2_eff = self.nonlinear_capacitor(v_OUT, C2, slope_c2) if self.nonlinear_C2 else C2

        L1_eff = self.nonlinear_inductor(i_IN, L1, slope_l1) if self.nonlinear_L1 else L1
        L2_eff = self.nonlinear_inductor(i_INT, L2, slope_l2) if self.nonlinear_L2 else L2

        # System equations
        d_v_INT_dt = (i_IN / C1_eff) - (i_INT / C1_eff)  #- (v_INT / (Rp1 * C1_eff))
        d_i_IN_dt = (v_IN / L1_eff) - ((R1 * i_IN) / L1_eff) - (v_INT / L1_eff)
        d_v_OUT_dt = (i_INT / C2_eff) - (i_OUT / C2_eff)  #- (v_OUT / (Rp2 * C2_eff))
        d_i_INT_dt = (v_INT / L2_eff) - ((R2 * i_INT) / L2_eff) - (v_OUT / L2_eff)

        return torch.stack([d_v_OUT_dt, d_i_IN_dt, d_v_INT_dt, d_i_INT_dt], dim=-1)


def generate_data(order=4,input_freqency=5,t_end=10,dt=0.001,scaling=1,flip_signal=False):
    if order != 4:
        raise ValueError("Only the fourth-order RLC dataset is supported.")

    if order==4:

            params = {'C1': 500 * 1e-6,
                      'L1': 90 * 1e-6,
                      'R1': 350 * 1e-3,
                      'Nonlinear_C1': True,
                      'Nonlinear_L1': True,
                      'Slope_C1': 0.05,
                      'Slope_L1': 0.5,
                      'C2': 200 * 1e-6,
                      'L2': 250 * 1e-6,
                      'R2': 200 * 1e-3,
                      'Nonlinear_C2': True,
                      'Nonlinear_L2': True,
                      'Slope_C2': 0.05,
                      'Slope_L2': 0.5}
            rlc_circuit = Order_4_RLC_Circuit(**params)


    t=np.linspace(0,t_end,int(t_end/dt))
    f = input_freqency  # Hz

    sig = (signal.square(2.1 * np.pi * f * t, duty=0.33)+1)/2
    if flip_signal:
        input=torch.tensor(np.array([sig,np.sin(2 * np.pi * f * t)])).T.unsqueeze(0)
    else:
        input=torch.tensor(np.array([np.sin(2 * np.pi * f * t),sig])).T.unsqueeze(0)
    input=input*scaling

    rlc_circuit.set_x(torch.tensor(t).unsqueeze(0),input)
    y_0=torch.zeros(1,order)
    with torch.no_grad():
        y=odeint(rlc_circuit, y_0, torch.tensor(t),
                 method='dopri5').permute(1,0,2)


    if order==4:
        df=pd.DataFrame(np.concatenate([t.reshape(-1,1),input.squeeze(0).numpy(),y.squeeze(0).numpy()],axis=1),columns=["time","V_in","I_out","V_out","I_in","V_int","I_int"])


    return df





class Order_4_Dataset(MultiFileTimeSeriesDataset):
    """
    Dataset that generates some data from the known ODE models
    """
    def __init__(self,config: dict, data_split:str='train',device: Optional[str] = 'cpu'):



        directory = Path(const.DATA_PATH, '4th_order_RLC',  data_split)

        if not directory.exists():
            directory.parent.mkdir(parents=True,exist_ok=True)
            train_data = generate_data(4, 300, 0.5, 0.00005, flip_signal=False)
            print("Train data generated")
            val_data = generate_data(4, 270, 0.1, 0.00005, flip_signal=False)
            print("Val data generated")
            test_data = generate_data(4, 250, 0.5, 0.00005, 0.9,flip_signal=False)
            print("Test data generated")
            path = Path(const.DATA_PATH, '4th_order_RLC', 'train')
            path.mkdir(exist_ok=True, parents=True)
            train_data.to_csv(path.joinpath("train_data.csv.gz"),index=False,compression='gzip')
            path = Path(const.DATA_PATH, '4th_order_RLC', 'val')
            path.mkdir(exist_ok=True, parents=True)
            val_data.to_csv(path.joinpath("val_data.csv.gz"),index=False,compression='gzip')
            path = Path(const.DATA_PATH, '4th_order_RLC', 'test')
            path.mkdir(exist_ok=True, parents=True)
            test_data.to_csv(path.joinpath("test_data.csv.gz"),index=False,compression='gzip')

            fig_train = px.line(train_data, x="time", y=["V_in", "I_out", "V_out", "I_in", "V_int", "I_int"],
                                title="Train data")
            fig_val = px.line(val_data, x="time", y=["V_in", "I_out", "V_out", "I_in", "V_int", "I_int"],
                              title="Val data")
            fig_test = px.line(test_data, x="time", y=["V_in", "I_out", "V_out", "I_in", "V_int", "I_int"],
                               title="Test data")

            path = Path(directory.parent, 'viz')
            path.mkdir(exist_ok=True, parents=True)
            fig_train.write_html(str(path.joinpath("train_data.html")))
            fig_val.write_html(str(path.joinpath("val_data.html")))
            fig_test.write_html(str(path.joinpath("test_data.html")))

        TRAIN_FILES = [directory.joinpath("train_data.csv.gz")]
        VAL_FILES = [directory.joinpath("val_data.csv.gz")]
        TEST_FILES = [directory.joinpath("test_data.csv.gz")]


        file_extension = ".csv.gz"
        path_data=Path(const.DATA_PATH, '4th_order_RLC')
        super().__init__(directory=path_data,
                         files=TRAIN_FILES if data_split == 'train' else VAL_FILES if data_split == 'val' else TEST_FILES,
                         config=config,
                         device=device,
                         data_split=data_split)

    def load_file(self, file_path) -> pd.DataFrame:
        data = pd.read_csv(file_path, delimiter=',',compression='gzip')
        return data




    @staticmethod
    def get_channels():
        time_channel = "time"
        x_channels = ["V_in", "I_out"]
        y_channels = ["V_out", "I_in","V_int", "I_int"]

        return {
            "time": time_channel,
            "excitation": x_channels,
            "response": y_channels,
            "auxiliary": []
        }




if __name__ == '__main__':




    print("Generating data for 4th order RLC circuit")
    train_data = generate_data(4, 200, 0.1, 0.00001, flip_signal=False)
    print("Train data generated")
    print("Val data generated")
    print("Test data generated")
    print("Data generated. Plot ant saving data")
    fig_train=px.line(train_data,x="time",y=["V_in","I_out","V_out","I_in","V_int","I_int"],title="Train data")

    path=Path(const.OUTPUT_PATH,'4th_Order_ds','viz')
    path.mkdir(exist_ok=True,parents=True)
    fig_train.write_html(str(path.joinpath("train_data.html")))



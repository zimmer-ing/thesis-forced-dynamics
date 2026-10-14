from pathlib import Path
import json
from src.datasets.dataset_base import MultiFileTimeSeriesDataset
from typing import Dict, Optional
import Constants as CONST
import torch
from scipy.integrate import odeint
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def generate_excitation_waveforms(trajectories_to_sample=100, t_nsamples=200, waveform_type="sawtooth", t_end=20.0, freq=0.5):
    """
    Generates various types of waveforms: 'sawtooth', 'sine', 'square', 'triangle', 'step', and more.

    Args:
        trajectories_to_sample (int): Number of trajectories to generate.
        t_nsamples (int): Number of time steps per trajectory.
        waveform_type (str): Type of waveform to generate.
        t_end (float): End time of the simulation.
        freq (float): Frequency of the waveform in Hz.

    Returns:
        torch.Tensor: Tensor of generated trajectories with shape [trajectories_to_sample, t_nsamples, 1].
        torch.Tensor: Time vector with shape [t_nsamples].
    """
    t_begin = t_end / t_nsamples
    ti = torch.linspace(t_begin, t_end, t_nsamples)

    x0s = torch.linspace(0, 2 * torch.pi, trajectories_to_sample)

    omega = 2 * torch.pi * freq

    def sawtooth_wave(t_in, x0=0):
        return (omega * t_in + x0) / (2 * torch.pi) - torch.floor((omega * t_in + x0) / (2 * torch.pi))

    def sine_wave(t_in, x0=0):
        return 0.5 * (1 + torch.sin(omega * t_in + x0))

    def square_wave(t_in, x0=0):
        return torch.sign(torch.sin(omega * t_in + x0)) * 0.5 + 0.5

    def triangle_wave(t_in, x0=0):
        return 2 * torch.abs(2 * ((omega * t_in + x0) / (2 * torch.pi) - torch.floor(0.5 + (omega * t_in + x0) / (2 * torch.pi)))) - 1

    def step_wave(t_in, x0=0, step_time=5.0):
        return (t_in >= step_time).float()

    def gaussian_wave(t_in, x0=0, sigma=1.0):
        return torch.exp(-((torch.sin(omega * t_in + x0)) ** 2) / (2 * sigma ** 2))

    def damped_sine_wave(t_in, x0=0, alpha=0.1):
        return torch.exp(-alpha * t_in) * torch.sin(omega * t_in + x0)

    def sigmoid_wave(t_in, x0=0):
        return 1 / (1 + torch.exp(-5 * torch.sin(omega * t_in + x0)))

    def exponential_wave(t_in, x0=0, alpha=0.1):
        return torch.sin(omega * t_in + x0) * torch.exp(-alpha * torch.abs(torch.sin(omega * t_in)))

    def hann_wave(t_in, x0=0):
        return 0.5 * (1 - torch.cos(omega * t_in / torch.pi))

    def blackman_wave(t_in, x0=0):
        return 0.42 - 0.5 * torch.cos(omega * t_in / torch.pi) + 0.08 * torch.cos(2 * omega * t_in / torch.pi)

    waveform_generators = {
        "sawtooth": sawtooth_wave,
        "sine": sine_wave,
        "square": square_wave,
        "triangle": triangle_wave,
        "step": step_wave,
        "gaussian": gaussian_wave,
        "damped_sine": damped_sine_wave,
        "sigmoid": sigmoid_wave,
        "exponential": exponential_wave,
        "hann": hann_wave,
        "blackman": blackman_wave,
    }

    if waveform_type not in waveform_generators:
        raise ValueError(f"Unsupported waveform type: {waveform_type}. Choose from {list(waveform_generators.keys())}.")

    generator = waveform_generators[waveform_type]

    trajs = [generator(ti, x0) for x0 in x0s]

    y = torch.stack(trajs)
    trajectories_generated = y.view(trajectories_to_sample, -1, 1)

    return trajectories_generated, ti

def smd_system(state, t, mass, damping, stiffness, forcing, time_forcing):
    """Differential equation of the spring-mass-damper system."""
    y, y_dot = state
    time_start = time_forcing[0]
    time_end = time_forcing[-1]
    forcing_value = np.interp(t, np.linspace(time_start, time_end, len(forcing)), forcing)
    dydt = [y_dot, (forcing_value - damping * y_dot - stiffness * y) / mass]
    return dydt


def generate_time_series(forcing_signal,time_forcing, initial_conditions, mass=1.0, damping=0.5, stiffness=5.0):
    """Generates time series data for a given forcing function and initial conditions."""

    #check if forcing signal is multidimensional -> if yes, assume multiple samples
    if len(forcing_signal.shape)>1:
        num_samples=forcing_signal.shape[0]
    else:
        num_samples=1
        #extend nd array with a leading dim
        forcing_signal=forcing_signal[None,:]
    time_series_data = []
    idx=0
    for i in range(num_samples):
        forcing_signal_i=forcing_signal[i]
        time_forcing_i=time_forcing

        for  init_cond in initial_conditions:
            y0, y_dot0 = init_cond
            initial_state = [y0, y_dot0]
            solution = odeint(smd_system, initial_state, time_forcing_i, args=(mass, damping, stiffness, forcing_signal_i,time_forcing_i))
            displacement, velocity = solution[:, 0], solution[:, 1]
            df = pd.DataFrame({
                "time": time_forcing_i,
                "forcing": forcing_signal_i,
                "displacement": displacement,
                "velocity": velocity,
                "y0": y0,
                "y_dot0": y_dot0
            })
            time_series_data.append((idx, df))
            idx+=1

    return time_series_data

class SMDSystem(MultiFileTimeSeriesDataset):



    def __init__(self, config: dict, data_split:str='train',device: Optional[str] = 'cpu'):
        if config["zero_initial_conditions"]:
            path_data = Path(CONST.DATA_PATH, "SMD_Dataset_ZeroIC")
        else:
            path_data = Path(CONST.DATA_PATH, "SMD_Dataset")
        t_nsamples = 550
        num_inital_conditions = 30
        excitations_num = {'train': 10,
                           'val': 5,
                           'test': 5}
        freqencies = {'train': 0.15,
                      'val': 0.15,
                      'test': 0.15}

        forcing_signals = {'train': "sigmoid",
                           'val': "damped_sine",
                           'test': "triangle"}

        # check if the data is already generated
        if not Path(path_data, 'train', 'Sample1.csv').exists():
            Path(path_data, 'train').mkdir(parents=True, exist_ok=True)
            Path(path_data, 'val').mkdir(parents=True, exist_ok=True)
            Path(path_data, 'test').mkdir(parents=True, exist_ok=True)

            # set seed to ensure reproducibility
            seed = torch.initial_seed()
            torch.manual_seed(seed)

            for ds_type in ['train', 'val', 'test']:
                waveform = forcing_signals[ds_type]
                freqency = freqencies[ds_type]
                excitation_num = excitations_num[ds_type]
                forcing_signal, time_forcing = generate_excitation_waveforms(trajectories_to_sample=excitation_num,
                                                                             t_nsamples=t_nsamples,
                                                                             waveform_type=waveform,
                                                                             freq=freqency)
                forcing_signal = forcing_signal.squeeze().numpy()
                time_forcing = time_forcing.numpy()
                initial_conditions = []
                if config["zero_initial_conditions"] :
                    initial_conditions=[[0.0,0.0]]
                else:
                    for _ in range(num_inital_conditions):
                        random_values = torch.rand(2).tolist()
                        initial_conditions.append(random_values)
                time_series_data = generate_time_series(forcing_signal, time_forcing, initial_conditions)
                for idx, df in time_series_data:
                    df.to_csv(Path(path_data, ds_type, f"Sample{idx}.csv"), index=False)
                fig = go.Figure()
                for idx, df in time_series_data:
                    fig.add_trace(go.Scatter(x=df["time"], y=df["displacement"], mode='lines', name=f"Sample {idx}"))
                    fig.add_trace(go.Scatter(x=df["time"], y=df["forcing"], mode='lines', name=f"Forcing Signal {idx}"))
                fig.update_layout(title="Spring-Mass-Damper System - Generated Time Series Data",
                                  xaxis_title="Time")
                fig.write_html(Path(path_data, ds_type, "Data.html"))

            #set seed back to original value
            torch.manual_seed(seed)

        if data_split not in ['train','val','test']:
            raise ValueError(f"Invalid data split: {data_split}. Must be one of ['train','val','test']")

        count = excitations_num[data_split] * (1 if config["zero_initial_conditions"] else num_inital_conditions)
        if config["zero_initial_conditions"]:
            filenames = [f"{data_split}/Sample{i}.csv" for i in range(count)]
        else:
            file_order = json.loads(Path(__file__).with_name("SMD_file_order.json").read_text())
            filenames = file_order[data_split]
        expected = {f"{data_split}/Sample{i}.csv" for i in range(count)}
        if len(filenames) != count or set(filenames) != expected:
            raise ValueError(f"Invalid SMD file order for {data_split}.")
        files = [path_data / name for name in filenames]

        super().__init__(directory=path_data,
                         files=files,
                         config=config,
                         device=device,
                         data_split=data_split)

    def load_file(self, file_path: Path) -> pd.DataFrame:
        return pd.read_csv(file_path, delimiter=",")

    @staticmethod
    def get_channels() -> Dict[str, str]:
        return {
            "time": "time",
            "excitation": [
                "forcing"
            ],
            "response": [
                "displacement",
            ],
            "auxiliary": []
        }

if __name__ == "__main__":
    import plotly.graph_objects as go
    import plotly.subplots as sp

    config = {
        "stride": 10,
        "history_length": 50,
        "forecast_length": 500,
        "batch_size": 32,
        "zero_initial_conditions": False
    }



    train_data = SMDSystem(config, data_split='train')
    val_data = SMDSystem(config, data_split='val')
    test_data = SMDSystem(config, data_split='test')

    def plot_sample(sample, sample_index=0):
        time_hist = sample[sample_index]["time_history"].numpy().flatten()
        time_fore = sample[sample_index]["time_forecast"].numpy().flatten()
        excitation_hist = sample[sample_index]["excitation_history"].numpy()
        excitation_fore = sample[sample_index]["excitation_forecast"].numpy()
        response_hist = sample[sample_index]["response_history"].numpy()
        response_fore = sample[sample_index]["response_forecast"].numpy()

        channels = SMDSystem.get_channels()
        excitation_channels = channels["excitation"]
        response_channels = channels["response"]

        fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Excitations", "Responses"))

        for i, channel_name in enumerate(excitation_channels):
            fig.add_trace(
                go.Scatter(x=time_hist, y=excitation_hist[:, i], mode='lines', name=f"{channel_name} History"),
                row=1, col=1)
            fig.add_trace(
                go.Scatter(x=time_fore, y=excitation_fore[:, i], mode='lines', name=f"{channel_name} Forecast"),
                row=1, col=1)

        for i, channel_name in enumerate(response_channels):
            fig.add_trace(
                go.Scatter(x=time_hist, y=response_hist[:, i], mode='lines', name=f"{channel_name} History"),
                row=1, col=2)
            fig.add_trace(
                go.Scatter(x=time_fore, y=response_fore[:, i], mode='lines', name=f"{channel_name} Forecast"),
                row=1, col=2)

        fig.update_layout(height=600, width=1200, title_text="SMD Dataset- Sample Visualization")
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Excitation Levels", row=1, col=1)
        fig.update_yaxes(title_text="Response Levels", row=1, col=2)

        fig.show()


    plot_sample(train_data)
    print("Train Data")

    #test inverse transform
    sample = train_data[0]
    excitation_fore = sample["excitation_forecast"]
    response_fore = sample["response_forecast"]
    excitation_hist = sample["excitation_history"]
    response_hist = sample["response_history"]


    excitation_fore_inv ,response_fore_inv= train_data.inverse_transform(excitation_fore, response_fore)
    excitation_hist_inv,response_hist_inv = train_data.inverse_transform(excitation_hist, response_hist)
    excitation = pd.concat((excitation_hist_inv, excitation_fore_inv), axis=0, ignore_index=True)
    response = pd.concat((response_hist_inv, response_fore_inv), axis=0, ignore_index=True)

    dir=Path(CONST.DATA_PATH, "SMD_Dataset")
    raw_data=pd.read_csv(dir / train_data.files[0])
    excitation_raw=raw_data[SMDSystem.get_channels()["excitation"]]
    response_raw=raw_data[SMDSystem.get_channels()["response"]]
    excitation_raw=excitation_raw.iloc[:config["history_length"]+config["forecast_length"]]
    response_raw=response_raw.iloc[:config["history_length"]+config["forecast_length"]]

    fig=sp.make_subplots(rows=1, cols=2, subplot_titles=("Excitations", "Responses"))
    for i, channel_name in enumerate(excitation_raw.columns):
        fig.add_trace(go.Scatter(x=excitation.index, y=excitation[channel_name], mode='markers', name=f"{channel_name} Transformed"),row=1, col=1)
        fig.add_trace(go.Scatter(x=excitation.index, y=excitation_raw[channel_name], mode='lines', name=f"{channel_name} Raw"),row=1, col=1)
    for i, channel_name in enumerate(response_raw.columns):
        fig.add_trace(go.Scatter(x=response.index, y=response[channel_name], mode='markers', name=f"{channel_name} Transformed"),row=1, col=2)
        fig.add_trace(go.Scatter
                        (x=response.index, y=response_raw[channel_name], mode='lines', name=f"{channel_name} Raw"),row=1, col=2)
    fig.show()

    # also check the first two samples for verification of the slicing
    sample1 = train_data[0]
    sample2 = train_data[1]
    excitation_fore1 = sample1["excitation_forecast"].numpy()
    response_fore1 = sample1["response_forecast"].numpy()
    excitation_hist1 = sample1["excitation_history"].numpy()
    response_hist1 = sample1["response_history"].numpy()
    excitation_fore2 = sample2["excitation_forecast"].numpy()
    response_fore2 = sample2["response_forecast"].numpy()
    excitation_hist2 = sample2["excitation_history"].numpy()
    response_hist2 = sample2["response_history"].numpy()
    channels = SMDSystem.get_channels()
    fig = sp.make_subplots(rows=2, cols=2, subplot_titles=("Excitations", "Responses"))
    for i in range(excitation_hist1.shape[1]):
        channel_name = channels["excitation"][i]
        index_array_hist = np.arange(len(excitation_hist1))
        index_array_fore = np.arange(len(excitation_fore1))+len(excitation_hist1)
        fig.add_trace(go.Scatter(x=index_array_hist, y=excitation_hist1[:, i], mode='lines', name=f"{channel_name} History 1"), row=1, col=1)
        fig.add_trace(go.Scatter(x=index_array_fore, y=excitation_fore1[:, i], mode='lines', name=f"{channel_name} Forecast 1"), row=1, col=1)
        fig.add_trace(go.Scatter(x=index_array_hist, y=excitation_hist2[:, i], mode='lines', name=f"{channel_name} History 2"), row=2, col=1)
        fig.add_trace(go.Scatter(x=index_array_fore, y=excitation_fore2[:, i], mode='lines', name=f"{channel_name} Forecast 2"), row=2, col=1)
    for i in range(response_hist1.shape[1]):
        channel_name = channels["response"][i]
        index_array_hist = np.arange(len(response_hist1))
        index_array_fore = np.arange(len(response_fore1)) +len(excitation_hist1)
        fig.add_trace(go.Scatter(x=index_array_hist, y=response_hist1[:, i], mode='lines', name=f"{channel_name} History 1"), row=1, col=2)
        fig.add_trace(go.Scatter(x=index_array_fore, y=response_fore1[:, i], mode='lines', name=f"{channel_name} Forecast 1"), row=1, col=2)
        fig.add_trace(go.Scatter(x=index_array_hist, y=response_hist2[:, i], mode='lines', name=f"{channel_name} History 2"), row=2, col=2)
        fig.add_trace(go.Scatter
                        (x=index_array_fore, y=response_fore2[:, i], mode='lines', name=f"{channel_name} Forecast 2"), row=2, col=2)
    fig.show()







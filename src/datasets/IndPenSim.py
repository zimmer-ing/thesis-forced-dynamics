import pandas as pd
from pathlib import Path
from src.datasets.dataset_base import MultiFileTimeSeriesDataset
from typing import Dict, Optional
import Constants as CONST
import numpy as np

class IndPenSimMinMax(MultiFileTimeSeriesDataset):

    TRAIN_FILES = [
        "Batch_1.csv",
        "Batch_2.csv",
        "Batch_3.csv",
        "Batch_4.csv",
        "Batch_5.csv",
        "Batch_6.csv",
        "Batch_7.csv",
        "Batch_8.csv",
        "Batch_9.csv",
        "Batch_10.csv",
        "Batch_30.csv",
        "Batch_31.csv",
        "Batch_32.csv",
        "Batch_33.csv",
        "Batch_34.csv",
        "Batch_35.csv",
        "Batch_36.csv",
        "Batch_37.csv",
        "Batch_38.csv",
        "Batch_39.csv",
        "Batch_40.csv",
        "Batch_61.csv",
        "Batch_62.csv",
        "Batch_63.csv",
        "Batch_64.csv",
        "Batch_65.csv",
        "Batch_66.csv",
        "Batch_67.csv",
        "Batch_68.csv",
        "Batch_69.csv",
        "Batch_70.csv",



    ]
    VAL_FILES = ["Batch_11.csv",
                "Batch_12.csv",
                 "Batch_41.csv",
                 "Batch_42.csv",
                 "Batch_71.csv",
                 "Batch_72.csv",]

    TEST_FILES = [
        "Batch_13.csv",
        "Batch_14.csv",
        "Batch_43.csv",
        "Batch_44.csv",
        "Batch_73.csv",
        "Batch_74.csv",

    ]

    def __init__(self, config: dict, data_split:str='train',device: Optional[str] = 'cpu'):
        if data_split not in ['train','val','test']:
            raise ValueError(f"Invalid data split: {data_split}. Must be one of ['train','val','test']")

        super().__init__(directory=Path(CONST.DATA_PATH, "IndPenSim", "minmax_scaled"),
                         files=self.TRAIN_FILES if data_split == 'train' else self.VAL_FILES if data_split == 'val' else self.TEST_FILES,
                         config=config,
                         device=device,
                         data_split=data_split)

    def load_file(self, file_path: Path) -> pd.DataFrame:
        return pd.read_csv(file_path, delimiter=",")

    @staticmethod
    def get_channels() -> Dict[str, str]:
        return {
            "time": "Time (h)",
            "excitation": [
                "Sugar feed rate(Fs:L/h)",
                "Acid flow rate(Fa:L/h)",
                "Base flow rate(Fb:L/h)",
                "Water for injection/dilution(Fw:L/h)",
                "PAA flow(Fpaa:PAA flow (L/h))",
                "Oil flow(Foil:L/hr)",
                "Aeration rate(Fg:L/h)",
                "Heating/cooling water flow rate(Fc:L/h)",
                "Heating water flow rate(Fh:L/h)",
                "Agitator RPM(RPM:RPM)",
                "Air head pressure(pressure:bar)"
            ],
            "response": [
                "Substrate concentration(S:g/L)",
                "Dissolved oxygen concentration(DO2:mg/L)",
                "Penicillin concentration(P:g/L)",
                "Vessel Volume(V:L)",
                "Vessel Weight(Wt:Kg)",
                "pH(pH:pH)",
                "Temperature(T:K)",
                "Generated heat(Q:kJ)",
                "carbon dioxide percent in off-gas(CO2outgas:%)",
                "Oxygen in percent in off-gas(O2:O2  (%))"
            ],
            "auxiliary": []
        }







from data_utils import candle_data_dict, Downloader
# from data_utils import load_gene_expression_data, load_drug_response_data, load_smiles_data
# from data_utils import add_smiles
import os
import pandas as pd
import numpy as np
import candle
import codecs
# from subword_nmt.apply_bpe import BPE
from download_model_data import download_model_specific_data
# from data_utils import load_landmark_genes
# from Step3_model import DeepTTC
import torch


file_path = os.path.dirname(os.path.realpath(__file__))
additional_definitions = [
    {'name': 'batch_size',
     'type': int
     },
    {'name': 'lr',
     'type': float,
     'help': 'learning rate'
     },
    {'name': 'epochs',
     'type': int
     },
    {'name': 'data_type',
     'type': str
     },
    {'name': 'data_split_seed',
     'type': int
     },
     {'name': 'metric',
     'type': str
     },
    {'name': 'download_data',
     'type': bool
     },
    {'name': 'data_source',
     'type': str
     },
    {'name': 'data_version',
     'type': str
     } 
]

required = None


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True

# setup_seed(0)


device_ids = [ int(os.environ["CUDA_VISIBLE_DEVICES"]) ]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CANDLE_DATA_DIR=os.getenv("CANDLE_DATA_DIR")


def run(params):

    # BATCH_SIZE = params['batch_size']
    num_epochs = params['epochs']
    output_dir = params['output_dir']
    download_data = params['download_data']
    metric = params['metric']
    data_path=os.path.join(CANDLE_DATA_DIR, params['model_name'], 'Data')
    data_source = params['data_source']
    data_type = candle_data_dict[data_source]
    data_split_id = params['data_split_id']
    data_split_seed = params['data_split_seed']
    data_version = params['data_version']

            

    downloadder = Downloader(data_version)
    downloadder.download_candle_data(data_type=data_type, split_id=data_split_id, data_dest=data_path)
    download_model_specific_data(data_dest=data_path)



class DeepTTC_candle(candle.Benchmark):

        def set_locals(self):
            if required is not None:
                self.required = set(required)
            if additional_definitions is not None:
                self.additional_definitions = additional_definitions

def initialize_parameters():
    """ Initialize the parameters for the GraphDRP benchmark. """
    print("Initializing parameters\n")
    swnet_params = DeepTTC_candle(
        filepath = file_path,
        defmodel = "deepttc_model.txt",
        framework = "pytorch",
        prog="DeepTTC",
        desc="CANDLE compliant DeepTTC",
    )
    gParameters = candle.finalize_parameters(swnet_params)
    return gParameters

if __name__ == '__main__':

    gParameters = initialize_parameters()
    run(gParameters)
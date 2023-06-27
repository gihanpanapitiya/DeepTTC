from data_utils import candle_data_dict, Downloader, DataProcessor
# from data_utils import load_gene_expression_data, load_drug_response_data, load_smiles_data
from data_utils import add_smiles
import os
import pandas as pd
import numpy as np
import candle
import codecs
from subword_nmt.apply_bpe import BPE
from download_model_data import download_model_specific_data
# from data_utils import load_landmark_genes
from Step3_model import DeepTTC
import torch
from sklearn.model_selection import train_test_split


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


# from Step1_getData import GetData
class DataEncoding:
    def __init__(self,vocab_dir, drug_smiles, metric, gexp, genes=None):
        self.vocab_dir = vocab_dir
        self.drug_smiles = drug_smiles
        self.metric = metric
        self.gexp = gexp
        
        if genes:
            self.gexp = self.gexp[genes]
        
        
        self.drugid2smile = dict(zip(drug_smiles['improve_chem_id'],drug_smiles['smiles']))
        smile_encode = pd.Series(drug_smiles['smiles'].unique()).apply(self._drug2emb_encoder)
        self.uniq_smile_dict = dict(zip(drug_smiles['smiles'].unique(),smile_encode))
        
        # self.Getdata = GetData()

    def _drug2emb_encoder(self,smile):
        vocab_path = "{}/drug_codes_chembl_freq_1500.txt".format(self.vocab_dir)
        sub_csv = pd.read_csv("{}/subword_units_map_chembl_freq_1500.csv".format(self.vocab_dir))

        bpe_codes_drug = codecs.open(vocab_path)
        dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

        idx2word_d = sub_csv['index'].values
        words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

        max_d = 50
        t1 = dbpe.process_line(smile).split()  # split
        try:
            i1 = np.asarray([words2idx_d[i] for i in t1])  # index
        except:
            i1 = np.array([0])

        l = len(i1)
        if l < max_d:
            i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (max_d - l))
        else:
            i = i1[:max_d]
            input_mask = [1] * max_d

        return i, np.asarray(input_mask)

    def encode(self, data_df):
        


        # data_df['smiles'] = [drugid2smile[i] for i in data_df['DRUG_ID']]
        # testdata['smiles'] = [drugid2smile[i] for i in testdata['DRUG_ID']]
        data_df['drug_encoding'] = [self.uniq_smile_dict[i] for i in data_df['smiles']]
        # testdata['drug_encoding'] = [uniq_smile_dict[i] for i in testdata['smiles']]
        data_df = data_df.reset_index()
        data_df['Label'] = data_df[ self.metric ]
        # testdata = testdata.reset_index()
        # testdata['Label'] = testdata[ self.metric ]

        # train_rnadata, test_rnadata = self.Getdata.getRna(
        #     traindata=traindata,
        #     testdata=testdata)
        # train_rnadata = train_rnadata.T
        # test_rnadata = test_rnadata.T
        
        rnadata = self.gexp.loc[data_df.improve_sample_id, :]
        rnadata.index = range(rnadata.shape[0])
        
        # test_rnadata.index = range(test_rnadata.shape[0])

        return data_df, rnadata
    

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

            
    if download_data:
        downloadder = Downloader(data_version)
        downloadder.download_candle_data(data_type=data_type, split_id=data_split_id, data_dest=data_path)
        download_model_specific_data(data_dest=data_path)

    data_processor = DataProcessor(data_version)

    train = data_processor.load_drug_response_data(data_path=data_path, data_type=data_type, split_id=data_split_id, split_type='train', response_type=metric)
    val = data_processor.load_drug_response_data(data_path=data_path, data_type=data_type, split_id=data_split_id, split_type='val', response_type=metric)
    test = data_processor.load_drug_response_data(data_path=data_path, data_type=data_type, split_id=data_split_id, split_type='test', response_type=metric)

    if data_split_seed> -1: # randomly shuffle data
        all_df = pd.concat([train, val, test])
        train, val = train_test_split(all_df, test_size=0.2, random_state=data_split_seed)
        test, val = train_test_split(val, test_size=0.5, random_state=data_split_seed)
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        val.reset_index(drop=True, inplace=True)

    smiles_df = data_processor.load_smiles_data(data_path)

    train = add_smiles(smiles_df=smiles_df, df=train, metric=metric)
    val = add_smiles(smiles_df=smiles_df, df=val, metric=metric)
    test = add_smiles(smiles_df=smiles_df, df=test, metric=metric)


    landmark_genes = data_processor.load_landmark_genes(data_path)
    gexp = data_processor.load_gene_expression_data(data_path)
    landmark_genes = list(set(gexp.columns).intersection(landmark_genes))
    input_dim_gene = len(landmark_genes)

    vocab_dir = os.path.join(data_path,'ESPF')
    enc  = DataEncoding(vocab_dir = vocab_dir, drug_smiles=smiles_df, metric=metric, gexp=gexp, genes=landmark_genes)

    train_df, train_rna = enc.encode(train)
    val_df, val_rna = enc.encode(val)
    test_df, test_rna = enc.encode(test)
    


    modelfile = output_dir + '/model.pt'
    # if not os.path.exists(modeldir):
    #     os.mkdir(modeldir)    

    net = DeepTTC(modeldir=output_dir, input_dim_gene = input_dim_gene)
    net.train(train_drug=train_df, train_rna=train_rna,
            val_drug=val_df, val_rna=val_rna, train_epoch=num_epochs)

    y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI = net.predict(test_df, test_rna)
    test_df['pred'] = y_pred
    test_df['true'] = y_label

    test_df.drop(['drug_encoding'], axis=1, inplace=True)
    test_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'))



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
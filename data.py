from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.loader import DataLoader
from data_process.dataSet import *
from data_process.loadData import *
import pickle
from torch.utils.data import DistributedSampler
"""
to run:
    nohup python -m torch.distributed.launch --nproc_per_node=3 --use_env parrel_new.py > train1216.log 2>&1 &
"""

def loadDrugCellData(args, mode, train_mode):

    if(mode == 'save'):
        print("start loading data from ./data ...")
        gexpr_feature, mutation_feature, methylation_feature, adeq_response, blind_response, drugid2smiles = loadDataFromFiles()
        aqed_dataset = MyDataset(gexpr_feature, mutation_feature, methylation_feature, adeq_response, drugid2smiles)
        blind_dataset = MyDataset(gexpr_feature, mutation_feature, methylation_feature, blind_response, drugid2smiles)
        train_dataset, dataset = train_test_split(aqed_dataset, test_size=0.2, random_state=args.seed)
        val_dataset, test_dataset = train_test_split(dataset, test_size=0.5, random_state=args.seed)

        # save
        # with open(f"./outputs/{args.exp_name}/train_dataset.pkl", "wb") as f:
        #     pickle.dump(train_dataset, f)
        #
        # with open(f"./outputs/{args.exp_name}/val_dataset.pkl", "wb") as f:
        #     pickle.dump(val_dataset, f)

        with open(f"./outputs/{args.exp_name}/test_dataset.pkl", "wb") as f:
            pickle.dump(test_dataset, f)

        print("Datasets saved successfully!")

    else:
        print("start load dataset...")
        with open("train_dataset.pkl", "rb") as f:
            train_dataset = pickle.load(f)

        with open("val_dataset.pkl", "rb") as f:
            val_dataset = pickle.load(f)

        with open("test_dataset.pkl", "rb") as f:
            test_dataset = pickle.load(f)

        with open("blind_dataset.pkl", "rb") as f:
            blind_dataset = pickle.load(f)
        print("finish load dataset!")

    train_sampler, val_loader, blind_loader= None, None, None
    if(train_mode == "single"):
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        blind_loader = DataLoader(blind_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    elif(train_mode == "multi"):
        train_loader = DataListLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataListLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    else:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset,
                                      batch_size=args.train_batch_size,
                                      sampler=train_sampler,
                                      num_workers=0,
                                      pin_memory=True)
        test_loader = DataLoader(test_dataset,
                                      batch_size=args.test_batch_size,
                                      num_workers=0,
                                      pin_memory=True)
    return train_loader, val_loader, test_loader, blind_loader, 21, 11, train_sampler

def getDrugBldDataLoader(args, mode, train_mode):

    print("start loading data from ./data ...")
    gexpr_feature, mutation_feature, methylation_feature, train_respond, test_respond, val_respond, drugid2smiles = loadDrugBlindTestDataFromFiles()
    train_dataset = MyDataset(gexpr_feature, mutation_feature, methylation_feature, train_respond, drugid2smiles)
    test_dataset = MyDataset(gexpr_feature, mutation_feature, methylation_feature, test_respond, drugid2smiles)
    val_dataset = MyDataset(gexpr_feature, mutation_feature, methylation_feature, val_respond, drugid2smiles)

    train_sampler, val_loader, blind_loader= None, None, None
    if(train_mode == "single"):
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)


    elif(train_mode == "multi"):
        train_loader = DataListLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataListLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    else:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset,
                                      batch_size=args.train_batch_size,
                                      sampler=train_sampler,
                                      num_workers=0,
                                      pin_memory=True)
        test_loader = DataLoader(test_dataset,
                                      batch_size=args.test_batch_size,
                                      num_workers=0,
                                      pin_memory=True)
    return train_loader, val_loader, test_loader, None, 21, 11, train_sampler

def getCellBldDataLoader(args, mode, train_mode):

    print("start loading data from ./data ...")
    gexpr_feature, mutation_feature, methylation_feature, train_respond, test_respond, val_respond, drugid2smiles = loadCellBlindTestDataFromFiles()
    train_dataset = MyDataset(gexpr_feature, mutation_feature, methylation_feature, train_respond, drugid2smiles)
    test_dataset = MyDataset(gexpr_feature, mutation_feature, methylation_feature, test_respond, drugid2smiles)
    val_dataset = MyDataset(gexpr_feature, mutation_feature, methylation_feature, val_respond, drugid2smiles)

    train_sampler, val_loader, blind_loader= None, None, None
    if(train_mode == "single"):
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)


    elif(train_mode == "multi"):
        train_loader = DataListLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataListLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    else:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset,
                                      batch_size=args.train_batch_size,
                                      sampler=train_sampler,
                                      num_workers=0,
                                      pin_memory=True)
        test_loader = DataLoader(test_dataset,
                                      batch_size=args.test_batch_size,
                                      num_workers=0,
                                      pin_memory=True)
    return train_loader, val_loader, test_loader, None, 21, 11, train_sampler

def loadExitAndNanDrugCellData(args, mode, train_mode):

    if(mode == 'save'):
        print("start loading data from ./data ...")
        gexpr_feature, mutation_feature, methylation_feature, exit_response, nan_response, drugid2smiles = loadExitAndNaDataFromFile()
        exit_dataset = MyDataset(gexpr_feature, mutation_feature, methylation_feature, exit_response, drugid2smiles)
        test_dataset = MyDataset(gexpr_feature, mutation_feature, methylation_feature, nan_response, drugid2smiles)
        train_dataset, val_dataset = train_test_split(exit_dataset, test_size=0.1, random_state=args.seed)

        #test_dataset, val_dataset = train_test_split(dataset, test_size=0.5, random_state=args.seed)
        # save
        # with open("train_dataset.pkl", "wb") as f:
        #     pickle.dump(train_dataset, f)
        #
        # with open("val_dataset.pkl", "wb") as f:
        #     pickle.dump(val_dataset, f)
        #
        # with open("test_dataset.pkl", "wb") as f:
        #     pickle.dump(test_dataset, f)
        #
        # with open("blind_dataset.pkl", "wb") as f:
        #     pickle.dump(blind_dataset, f)
        print("Datasets saved successfully!")

    else:
        print("start load dataset...")
        with open("train_dataset.pkl", "rb") as f:
            train_dataset = pickle.load(f)

        with open("val_dataset.pkl", "rb") as f:
            val_dataset = pickle.load(f)

        with open("test_dataset.pkl", "rb") as f:
            test_dataset = pickle.load(f)

        with open("blind_dataset.pkl", "rb") as f:
            blind_dataset = pickle.load(f)
        print("finish load dataset!")

    train_sampler, val_loader, blind_loader= None, None, None
    if(train_mode == "single"):
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    elif(train_mode == "multi"):
        train_loader = DataListLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataListLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    else:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset,
                                      batch_size=args.train_batch_size,
                                      sampler=train_sampler,
                                      num_workers=0,
                                      pin_memory=True)
        test_loader = DataLoader(test_dataset,
                                      batch_size=args.test_batch_size,
                                      num_workers=0,
                                      pin_memory=True)
    return train_loader, val_loader, test_loader, blind_loader, 21, 11, train_sampler

def loadAllBalanceData(batch_size):
    print("start loading data from ./data ...")
    gexpr_feature, mutation_feature, methylation_feature, adeq_response, blind_response, drugid2smiles = loadDataFromFiles()
    dataset = MyDataset(gexpr_feature, mutation_feature, methylation_feature, adeq_response, drugid2smiles)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


# 测试代码
"""
class Args:
    def __init__(self, seed=42):
        self.seed = seed
        self.train_batch_size = 128
        self.test_batch_size = 128
args = Args()
loadDrugCellData(args,"load","single")
"""

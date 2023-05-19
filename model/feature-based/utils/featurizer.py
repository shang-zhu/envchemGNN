import pandas as pd
import numpy as np
import os 
import pickle as pkl
from rdkit import Chem 
from rdkit.Chem import AllChem, MACCSkeys
from mordred import Calculator, descriptors


def compute_rdkitfp(smiles: str, fp='ecfp'):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as E:
        return None

    if mol:
        if fp=='ecfp':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)        
        elif fp=='maccs':
            fp =MACCSkeys.GenMACCSKeys(mol)
        
        return np.array(fp).reshape((1, -1))
    return None

def smi2feat(lcia_csv, feat_dir, feat_type='ecfp'):
    lcia_df=pd.read_csv(lcia_csv)[['SMILES']]
    smi_list=list(lcia_df['SMILES'])
    if feat_type=='grover':
        dir=feat_dir+'grover/'
        model_path='/home/shangzhu/projects/molecules/pretainLCA/grover/model/grover_large.pt'
        if not os.path.exists(dir):
            os.makedirs(dir)
        lcia_df.to_csv(dir+"smi.csv", index=False)
        string1="python ../grover/scripts/save_features.py --data_path "+ dir +"smi.csv \
                                    --save_path "+ dir+"data.npz \
                                    --features_generator morgan \
                                    --restart"
        string2="python ../grover/main.py fingerprint --data_path "+ dir +"smi.csv \
            --features_path "+ dir+"data.npz \
            --checkpoint_path "+ model_path+" \
            --fingerprint_source both \
            --output "+ dir+"feat.npz"
        os.system(string1)
        print('grover vocab generated...')
        os.system(string2)
        print("GROVER pretraining features generated at"+ dir + "feat.npz (np.load(*npz)['fps'])")
        np.save(dir+'feat.npy', np.load(dir+'feat.npz',allow_pickle=True)['fps'])
    elif feat_type=='ecfp':
        dir=feat_dir+'ecfp/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        molfps=np.empty([0, 2048])
        for idx, mol in enumerate(smi_list):
            try:
                molfp = compute_rdkitfp(mol, feat_type)
                molfps=np.concatenate((molfps,molfp), axis=0)
            except:
                pass    
        np.save(dir+'feat.npy', molfps)
        print('ECFP features generated at data/ecfp/feat.npy')

    elif feat_type=='maccs':
        dir=feat_dir+'maccs/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        molfps=np.empty([0, 167])
        for idx, mol in enumerate(smi_list):
            try:
                molfp = compute_rdkitfp(mol, feat_type)
                molfps=np.concatenate((molfps,molfp), axis=0)
            except:
                pass    
        np.save(dir+'feat.npy', molfps)
        print('MACCS features generated at data/maccs/feat.npy')

    elif feat_type=='mordred':
        dir=feat_dir+'mordred/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        #only 2d features can be accessed with SMILES 
        calc2d = Calculator(descriptors, ignore_3D=True)
        mols = [Chem.MolFromSmiles(smi) for smi in smi_list]
        calc_df = calc2d.pandas(mols)
        calc_npy=calc_df.to_numpy().astype(dtype=float)
        np.save(dir+'feat.npy', calc_npy[:,~np.isnan(calc_npy).any(axis=0)])
        np.save(dir+'name.npy', np.array([~np.isnan(calc_npy).any(axis=0)]))
        print('Mordred 2D features generated at data/mordred/feat.npy')
    elif feat_type=='dvmp':
        #this feat is generated externally due to the integration difficulty
        dir=feat_dir+'dvmp_ext/'
        if not os.path.exists(dir):
            print('no preprocessed data found')
        else:
            fr=open(dir+"feat.pkl", 'rb')
            feat=pkl.load(fr)
            trans_dir=feat_dir+'dvmp_trans/'
            gnn_dir=feat_dir+'dvmp_gnn/'
            if not os.path.exists(trans_dir):
                os.makedirs(trans_dir)
            np.save(trans_dir+'feat.npy', feat[0])
            if not os.path.exists(gnn_dir):
                os.makedirs(gnn_dir)
            np.save(gnn_dir+'feat.npy', feat[1])

            print('DVMP transformer features generated at data/dvmp_trans/feat.npy')
            print('DVMP gnn features generated at data/dvmp_gnn/feat.npy')

def get_mordred_with_name(lcia_csv, feat_dir, source_dir):
    lcia_df=pd.read_csv(lcia_csv)[['SMILES']]
    smi_list=list(lcia_df['SMILES'])

    dir=feat_dir+'mordred/'
    valid_index=np.load(source_dir+'name.npy').reshape(-1)
    if not os.path.exists(dir):
        os.makedirs(dir)
    #only 2d features can be accessed with SMILES 
    calc2d = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in smi_list]
    calc_df = calc2d.pandas(mols)
    calc_npy=calc_df.iloc[:, valid_index].to_numpy().astype(dtype=float)
    np.save(dir+'feat.npy', calc_npy)#[:,~np.isnan(calc_npy).any(axis=0)]
    print('Mordred 2D features generated at '+feat_dir+'feat.npy')
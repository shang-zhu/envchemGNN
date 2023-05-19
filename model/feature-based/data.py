from utils.featurizer import smi2feat
import argparse, os
 
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)

args = parser.parse_args()
label_csv=args.input_path
feat_dir=args.output_path

#creating results folder
if not os.path.exists(feat_dir):
    os.makedirs(feat_dir)
    
#'grover' 'dvmp' features are left for a future study
for feat_type in ['maccs', 'ecfp','mordred']:
    smi2feat(label_csv, feat_dir, feat_type=feat_type)
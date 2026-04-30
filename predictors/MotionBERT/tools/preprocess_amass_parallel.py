import torch
import torch.multiprocessing as mp
import numpy as np
import os
import sys
import argparse

here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, ".."))

from os import path as osp
from human_body_prior.body_model.body_model import BodyModel
import pickle
import pandas as pd
from functools import partial


def process_motion(args):
    """Process a single motion sequence and return the joints data."""
    idx, bdata, fname, J_reg, max_len = args
    
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_gender = bdata['gender']
    if (str(subject_gender) != 'female') and (str(subject_gender) != 'male'):
        subject_gender = 'female'

    bm_fname = osp.join('data/AMASS/body_models/smplh/{}/model.npz'.format(subject_gender))
    dmpl_fname = osp.join('data/AMASS/body_models/dmpls/{}/model.npz'.format(subject_gender))

    # number of body parameters
    num_betas = 16
    # number of DMPL parameters
    num_dmpls = 8

    bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)
    time_length = len(bdata['trans'])
    num_slice = time_length // max_len

    results = []
    for sid in range(num_slice + 1):
        start = sid * max_len
        end = min((sid + 1) * max_len, time_length)
        body_parms = {
            'root_orient': torch.Tensor(bdata['poses'][start:end, :3]).to(comp_device),
            'pose_body': torch.Tensor(bdata['poses'][start:end, 3:66]).to(comp_device),
            'pose_hand': torch.Tensor(bdata['poses'][start:end, 66:]).to(comp_device),
            'trans': torch.Tensor(bdata['trans'][start:end]).to(comp_device),
            'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=(end - start), axis=0)).to(comp_device),
            'dmpls': torch.Tensor(bdata['dmpls'][start:end, :num_dmpls]).to(comp_device)
        }
        body_trans_root = bm(**{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls', 'trans', 'root_orient']})
        mesh = body_trans_root.v.cpu().numpy()
        kpts = np.dot(J_reg, mesh)  # (17,T,3)
        results.append({
            'idx': idx,
            'sid': sid,
            'kpts': kpts,
            'fname': fname,
            'clip_len': end - start
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Preprocess AMASS dataset with parallel processing')
    parser.add_argument('--num_workers', type=int, default=1, 
                        help='Number of parallel workers (default: 1)')
    args = parser.parse_args()

    # Set multiprocessing start method for CUDA compatibility
    if args.num_workers > 1:
        mp.set_start_method('spawn', force=True)

    df = pd.read_csv('./data/AMASS/fps.csv', sep=',', header=None)
    fname_list = list(df[0][1:])

    J_reg_dir = 'data/AMASS/J_regressor_h36m_correct.npy'
    all_motions = 'data/AMASS/all_motions_fps60.pkl'

    with open(all_motions, 'rb') as file:
        motion_data = pickle.load(file)
    
    J_reg = np.load(J_reg_dir)
    max_len = 2916

    # Prepare arguments for parallel processing
    process_args = [
        (i, bdata, fname_list[i], J_reg, max_len) 
        for i, bdata in enumerate(motion_data)
    ]

    total = len(motion_data)
    print(f'Processing {total} sequences with {args.num_workers} worker(s)...')

    if args.num_workers == 1:
        # Sequential processing
        all_results = []
        for i, arg in enumerate(process_args):
            if i % 200 == 0:
                print(f'{i} seqs done. (total: {total})')
            results = process_motion(arg)
            all_results.extend(results)
    else:
        # Parallel processing
        all_results = []
        with mp.Pool(processes=args.num_workers) as pool:
            for i, results in enumerate(pool.imap(process_motion, process_args)):
                if i % 200 == 0:
                    print(f'{i} seqs done. (total: {total})')
                all_results.extend(results)

    # Sort results by original index and slice id to maintain order
    all_results.sort(key=lambda x: (x['idx'], x['sid']))

    # Write clip list CSV and collect joints
    all_joints = []
    with open('data/AMASS/clip_list.csv', 'w') as f:
        print('clip_id, fname, clip_len', file=f)
        for clip_id, result in enumerate(all_results):
            all_joints.append(result['kpts'])
            print(f"{clip_id}, {result['fname']}, {result['clip_len']}", file=f)

    # Save joints
    with open('data/AMASS/amass_joints_h36m_60.pkl', 'wb') as fileName:
        pickle.dump(all_joints, fileName)
    
    print(f'Done! Total clips: {len(all_joints)}')


if __name__ == '__main__':
    main()


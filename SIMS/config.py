import torch
import os

seed = 1111
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
root_path = os.path.dirname(__file__)

LOGPATH = os.path.join(root_path, 'log/')

if not os.path.exists(LOGPATH): 
    os.makedirs(LOGPATH)
                
keys = ['T{}s', 'V{}s', 'A{}s', 'T{}d', 'V{}d', 'A{}d']
num_sample = 7
all_keys = {}
idx = 0
for item in keys:
    for i in range(num_sample):
        all_keys[(item.replace('{}', str(i)))] = idx
        idx += 1
positive_pairs = [
    # inter-sample pairing
    'T0s,T1s',
    'T0s,T2s',
    'V0s,V1s',
    'V0s,V2s',
    'A0s,A1s',
    'A0s,A2s',
    # intra-sample pairing
    'T0s,V0s',
    'T0s,A0s',
    'T1s,V1s',
    'T1s,A1s',
    'T2s,V2s',
    'T2s,A2s',
    'T3s,V3s',
    'T3s,A3s',
    'T4s,V4s',
    'T4s,A4s',
    'T5s,V5s',
    'T5s,A5s',
    'T6s,V6s',
    'T6s,A6s',
]
negative_pairs = [
    # inter-sample pairing
    'T0s,T3s',
    'T0s,T4s',
    'T0s,T5s',
    'T0s,T6s',
    'V0s,V3s',
    'V0s,V4s',
    'V0s,V5s',
    'V0s,V6s',
    'A0s,A3s',
    'A0s,A4s',
    'A0s,A5s',
    'A0s,A6s',
    # intra-sample pairing
    'T0s,T0d',
    'T0s,V0d',
    'T0s,A0d',
    'T1s,T1d',
    'T1s,V1d',
    'T1s,A1d',
    'T2s,T2d',
    'T2s,V2d',
    'T2s,A2d',
    'T3s,T3d',
    'T3s,V3d',
    'T3s,A3d',
    'T4s,T4d',
    'T4s,V4d',
    'T4s,A4d',
    'T5s,T5d',
    'T5s,V5d',
    'T5s,A5d',
    'T6s,T6d',
    'T6s,V6d',
    'T6s,A6d',
]

t1, p, t2, n = [], [], [], []
for pair in positive_pairs:
    eA, eB = pair.split(',')
    eA_idx = all_keys[eA]
    eB_idx = all_keys[eB]
    t1.append(eA_idx)
    p.append(eB_idx)
for pair in negative_pairs:
    eA, eB = pair.split(',')
    eA_idx = all_keys[eA]
    eB_idx = all_keys[eB]
    t2.append(eA_idx)
    n.append(eB_idx)


class SIMS:
    class path:
        raw_data_path = '../Datasets/SIMS/Processed/unaligned_39.pkl'
        model_path = os.path.join(root_path, './save_models/all_model/SIMS/')
        if not os.path.exists(model_path): 
            os.makedirs(model_path)
        encoder_path = os.path.join(root_path, './save_models/uni_fea_encoder/SIMS/')
        if not os.path.exists(encoder_path): 
            os.makedirs(encoder_path)

    class downStream:
        language = 'cn'
      
        encoder_fea_dim = 768
        
        text_fea_dim = 768
        
        vision_fea_dim = 709
        vision_seq_len = 55
        
        audio_fea_dim = 33
        audio_seq_len = 400
        
        vision_drop_out, audio_drop_out = 0.5, 0.5
        vision_nhead, audio_nhead = 8, 8
        vision_dim_feedforward = encoder_fea_dim 
        audio_dim_feedforward = encoder_fea_dim 
        vision_tf_num_layers, audio_tf_num_layers = 5, 3
        vision_attn_mask, audio_attn_mask = True, True
        
        audio_text_nhead, vision_text_nhead = 8, 8
        vision_text_tf_num_layers, audio_text_tf_num_layers = 5, 2
        drop_out = 0.6
        text_drop_out = 0.5
        attn_mask = True

        batch_size = 32
        update_epochs = 4
        
        alen=audio_seq_len
        vlen=vision_seq_len 
        p_len= 3
        
        class visionPretrain:
            lr = 1e-4
            epoch = 25
            decay = 1e-3
         
        class audioPretrain:
            lr = 1e-3
            epoch = 25
            decay = 1e-3
      
        class TVAtrain:	
            text_lr = 5e-5
            audio_lr = 1e-3
            vision_lr = 1e-3
            other_lr = 1e-3
            
            text_decay = 1e-3
            audio_decay = 1e-3
            vision_decay = 1e-2
            other_decay = 1e-2
            
            epoch = 50
            
            delta_va = 0.5
            delta_tva = 0.5
            delta_nce = 0.005
           

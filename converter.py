from collections import OrderedDict as OrderedDict

import torch
import numpy as np

from tqdm import tqdm


def convert(npz):
    state_dict = OrderedDict()
    state_dict['cls_token'] = torch.tensor(npz['cls'])
    size = -1
    try:
        for i in tqdm(npz):
            if i == 'cls' or 'key' in i or 'value' in i:
                continue
            key = reanme(i)
            value = torch.tensor(npz[i])
            if 'fc' in key or 'head' in key:
                value = value.T
            elif 'att' in key:
                if 'bias' in key:
                    value = value.view(-1)
                    size = value.shape[0]
                    if 'qkv' in key:
                        value = torch.cat((value, torch.tensor(npz[i.replace('query', 'key')]).view(-1).T, torch.tensor(npz[i.replace('query', 'value')]).view(-1).T))
                if 'weight' in key:
                    value = value.view(-1, size).T
                    if 'qkv' in key:
                        value = torch.cat((value, torch.tensor(npz[i.replace('query', 'key')]).view(-1, size).T, torch.tensor(npz[i.replace('query', 'value')]).view(-1, size).T))
            elif 'patch_embed.proj.weight' in key:
                value = value.permute(3, 2, 0, 1)
            if key != 'pos_embed':
                value = value.squeeze()
            state_dict[key] = value
    except Exception:
         import pdb; pdb.set_trace()
    return state_dict


def reanme(s):
    s = s.replace('Transformer/', '').replace('encoder_', '')
    s = s.replace('_', '.').replace('/', '.')
    s = s.replace('scale', 'weight').replace('kernel', 'weight')
    s = s.replace('LayerNorm', 'norm').replace('MlpBlock', 'mlp')
    s = s.replace('MultiHeadDotProductAttention.1', 'attention')
    # s = s.replace('MultiHeadDotProductAttention.1', 'attn')
    s = s.replace('encoderblock', 'blocks')
    s = s.replace('norm.0', 'norm1').replace('norm.2', 'norm2')
    s = s.replace('3.Dense.0', 'fc1').replace('3.Dense.1', 'fc2')
    s = s.replace('posembed.input.pos.embedding', 'pos_embed')
    s = s.replace('embedding', 'patch_embed.proj') 
    s = s.replace('out', 'proj').replace('query', 'qkv')
    return s


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert JFX .npz to PyTorch .pth')
    parser.add_argument('-s', '--source', type=str, required=True)
    parser.add_argument('-t', '--target', type=str, default=None)
    args = parser.parse_args()
    if not args.source.endswith('.npz'):
        raise ValueError('Invalid source file provided, it must ends with .npz')
    if not args.target:
        args.target = args.source.replace('.npz', '.pth')
    npz = np.load(args.source)
    sd = convert(npz)
    torch.save(sd, args.target)

import torch.optim
from torch.utils import data
import argparse
import json
from tqdm import tqdm
from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
from data.DUBAI_CC.DUBAICC import DUBAICCDataset
from data.BD_CC.BDCC import BDCCDataset

from model.encoder import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *

def main(args):
    """
    Testing.
    """
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    # Load checkpoint
    snapshot_full_path = args.checkpoint
    checkpoint = torch.load(snapshot_full_path)

    encoder = Encoder(args.network, args.data_name, args.embed_dim, args.encoder_dim, args.prompt_type)
    encoder_trans = AttentiveEncoder(n_layers=args.n_layers,
                                          feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                                          heads=args.n_heads, dropout=args.dropout)
    decoder = DecoderTransformer(decoder_type=args.decoder_type,embed_dim=args.encoder_dim,
                                      vocab_size=len(word_vocab), max_lengths=args.max_length,
                                      word_vocab=word_vocab, n_head=args.n_heads,
                                      n_layers=args.decoder_n_layers, dropout=args.dropout)
    

    encoder.load_state_dict(checkpoint['encoder_dict'])
    encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'])
    decoder.load_state_dict(checkpoint['decoder_dict'])
    # Move to GPU, if available
    encoder.eval()
    encoder = encoder.cuda()
    encoder_trans.eval()
    encoder_trans = encoder_trans.cuda()
    decoder.eval()
    decoder = decoder.cuda()
    print('load model success!')

    if args.data_name == 'LEVIR_CC':
        test_loader = data.DataLoader(
                LEVIRCCDataset(args.network, args.data_folder, args.list_path, 'test', args.token_folder, word_vocab, args.max_length, args.allow_unk),
                batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.data_name == 'DUBAI_CC':
        test_loader = data.DataLoader(
            DUBAICCDataset(args.network, args.data_folder, args.list_path, 'test', args.token_folder, word_vocab, args.max_length, args.allow_unk),
                batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

    elif args.data_name == 'BD_CC':
        test_loader = data.DataLoader(
            BDCCDataset(args.network, args.data_folder, args.list_path, 'test', args.token_folder, word_vocab, args.max_length, args.allow_unk),
                batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

    with torch.no_grad():
        # Batches
        with open('bd_output.txt', 'w') as file:
            for ind, batch_data in enumerate(
                    tqdm(test_loader, desc='test_' + " EVALUATING AT BEAM SIZE " + str(1))):
                # Move to GPU, if available
                imgA = batch_data['imgA']
                imgB = batch_data['imgB']
                name = batch_data['name']
                imgA = imgA.cuda()
                imgB = imgB.cuda()
                # Forward prop.
                if encoder is not None:
                    feat1, feat2 = encoder(imgA, imgB)
                feat = encoder_trans(feat1, feat2)
                seq = decoder.sample(feat, k=1)

                # for captioning
                except_tokens = {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}
                pred_seq = [w for w in seq if w not in except_tokens]
                pred_caption = ""
                for i in pred_seq:
                    pred_caption += (list(word_vocab.keys())[i]) + " "

                file.write(f'{name[0]}:{pred_caption}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Captioning')

    # Data parameters
    parser.add_argument('--sys', default='linux', choices=('linux'), help='system')
    # parser.add_argument('--data_folder',
    #                     default='/home/september/code/ICC/data/LEVIR_CC/images',
    #                     help='folder with data files')
    
    # parser.add_argument('--list_path',
    #                     default='./data/LEVIR_CC/',
    #                     help='path of the data lists')
    
    # parser.add_argument('--token_folder',
    #                     default='./data/LEVIR_CC/tokens/',
    #                     help='folder with token files')
    
    # parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    # parser.add_argument('--max_length', type=int, default=42, help='path of the data lists')
    # parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    # parser.add_argument('--data_name', default="LEVIR_CC", help='base name shared by data files.')


    # parser.add_argument('--data_folder',
    #                     default='/home/september/code/ICC/data/DUBAI_CC/images',
    #                     help='folder with data files')
    #
    # parser.add_argument('--list_path',
    #                     default='./data/DUBAI_CC/',
    #                     help='path of the data lists')
    #
    # parser.add_argument('--token_folder',
    #                     default='./data/DUBAI_CC/tokens/',
    #                     help='folder with token files')
    #
    # parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    # parser.add_argument('--max_length', type=int, default=27, help='path of the data lists')
    # parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    # parser.add_argument('--data_name', default="DUBAI_CC", help='base name shared by data files.')


    parser.add_argument('--data_folder',
                        default='/home/september/code/ICC/data/BD_CC/images',
                        help='folder with data files')

    parser.add_argument('--list_path',
                        default='./data/BD_CC/',
                        help='path of the data lists')

    parser.add_argument('--token_folder',
                        default='./data/BD_CC/tokens/',
                        help='folder with token files')

    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=63, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="BD_CC", help='base name shared by data files.')

    # Test
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')

    parser.add_argument('--checkpoint',
                        default='result/BD_CC/soft_keywords/BD_CC_bts_64_RemoteCLIP-ViT-B-32_epo_0_0.pth',
                        help='path to checkpoint, None if none.')
    parser.add_argument('--prompt_type', default="soft_keywords")

    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')
    parser.add_argument('--test_batchsize', default=1, help='batch_size for validation')
    parser.add_argument('--workers', type=int, default=8,
                        help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # Validation
    parser.add_argument('--result_path', default="result/out")

    # backbone parameters
    parser.add_argument('--decoder_type', default='transformer_decoder', help='mamba or gpt or transformer_decoder')
    parser.add_argument('--network', default='RemoteCLIP-ViT-B-32',help='define the backbone encoder to extract features')
    # parser.add_argument('--encoder_dim', type=int, default=768, help='the dim of extracted features of backbone ')
    # parser.add_argument('--feat_size', type=int, default=16, help='size of extracted features of backbone')
    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers in AttentionEncoder.')
    parser.add_argument('--decoder_n_layers', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=768, help='embedding dimension')
    args = parser.parse_args()

    print('list_path:', args.list_path)
    
    if args.network == 'CLIP-ViT-B/16':
        args.embed_dim, args.encoder_dim, args.feat_size = 512, 768, 14
    elif args.network == 'CLIP-ViT-B/32':
        args.embed_dim, args.encoder_dim, args.feat_size = 512, 768, 7
    elif args.network == 'RemoteCLIP-ViT-B-32':
        args.embed_dim, args.encoder_dim, args.feat_size = 512, 768, 7
    elif args.network == 'RemoteCLIP-ViT-L-14':
        args.embed_dim, args.encoder_dim, args.feat_size = 768, 1024, 16
    elif args.network == 'alexnet':
        args.encoder_dim, args.feat_size = 256, 7
    elif args.network == 'vgg19':
        args.encoder_dim, args.feat_size = 512, 8
    elif args.network == 'resnet18':
        args.encoder_dim, args.feat_size = 512, 8
    elif args.network == 'resnet34':
        args.encoder_dim, args.feat_size = 512, 8
    elif args.network == 'resnet50':
        args.encoder_dim, args.feat_size = 2048, 8
    elif args.network == 'resnet101':
        args.encoder_dim, args.feat_size = 2048, 14
    else:
        print('Backbone Error!')
        exit()

    main(args)

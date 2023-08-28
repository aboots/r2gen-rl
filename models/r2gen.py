import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        elif args.dataset_name == 'mimic_cxr':
            self.forward = self.forward_mimic_cxr
        else:
            self.forward = self.forward_ffa_ir

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

        
    # def forward_ffa_ir(self, images, targets=None, mode='train', update_opts={}):
    #     print(images.shape)
    #     att_feats_list = []
    #     fc_feats_list = []
    #     for i in range(images.shape[1]):
    #         att_feats_i, fc_feats_i = self.visual_extractor(images[:, i,:,:])
    #         att_feats_list.append(att_feats_i)
    #         fc_feats_list.append(fc_feats_i)
    #     fc_feats = torch.cat(fc_feats_list, dim=1)
    #     att_feats = torch.cat(att_feats_list, dim=1)


    #     if mode == 'train':
    #         output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
    #         return output
    #     elif mode == 'sample':
    #         output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample')
    #         return output, output_probs
    #     else:
    #         raise ValueError

    def forward_ffa_ir(self, images, targets=None, mode='train', update_opts={}):
        att_feats_list = []
        fc_feats_list = []

        for i in range(images.shape[1]):
            att_feats_i, fc_feats_i = self.visual_extractor(images[:, i,:,:])
            att_feats_list.append(att_feats_i)
            fc_feats_list.append(fc_feats_i)
        avg_att_feats_1 = torch.mean(torch.stack(att_feats_list[:30]), dim=0)
        avg_att_feats_2 = torch.mean(torch.stack(att_feats_list[30:]), dim=0)
        avg_fc_feats_1 = torch.mean(torch.stack(fc_feats_list[:30]), dim=0)
        avg_fc_feats_2 = torch.mean(torch.stack(fc_feats_list[30:]), dim=0)
        att_feats = torch.cat([avg_att_feats_1, avg_att_feats_2], dim=1)
        fc_feats = torch.cat([avg_fc_feats_1, avg_fc_feats_2], dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            return output
        else:
            raise ValueError

    # def forward_ffa_ir(self, images, targets=None, mode='train'):
    #     att_feats = 0
    #     fc_feats = 0
    #     print(images.shape)
    #     for ind in range(images.shape[1]):
    #         att_feats_new, fc_feats_new = self.visual_extractor(images[:, ind])
    #         att_feats += att_feats_new
    #         fc_feats += fc_feats_new
    #     att_feats /= images.shape[1]
    #     fc_feats /= images.shape[1]
    #     if mode == 'train':
    #         output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
    #     elif mode == 'sample':
    #         output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
    #     else:
    #         raise ValueError
    #     return output

    # def forward_ffa_ir(self, images, targets=None, mode='train', update_opts={}):
    #     print(f'hi2 {images.shape}')
    #     for i in range(images.shape[1]):
    #         att_features, fc_features =  self.visual_extractor(images[i])
    #         if i == 0:
    #             att_feats = att_features.copy()
    #             fc_feats = fc_features.copy()
    #         else:
    #             att_feats += att_features
    #             fc_feats += fc_features
    #     fc_feats /= images.shape[1]
    #     att_feats /= images.shape[1]
    #     if mode == 'train':
    #         output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
    #     elif mode == 'sample':
    #         output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
    #     else:
    #         raise ValueError
    #     return output
    
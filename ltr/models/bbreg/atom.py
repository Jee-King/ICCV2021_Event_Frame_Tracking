import torch.nn as nn
import ltr.models.backbone as backbones
import ltr.models.bbreg as bbmodels
from ltr import model_constructor
# import ltr.models.backbone.inter_attention as inter_atten


class ATOMnet(nn.Module):
    """ ATOM network module"""
    def __init__(self, feature_extractor, bb_regressor, bb_regressor_layer, extractor_grad=True):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(ATOMnet, self).__init__()

        self.feature_extractor = feature_extractor
        self.bb_regressor = bb_regressor
        self.bb_regressor_layer = bb_regressor_layer
        # self.inter_atten1 = backbones.Cha_Spa()
        # self.inter_atten2 = backbones.Cha_Spa()
        self.motion_attention = backbones.Motion_Attention()
        self.counter_guide = backbones.Counter_Guide()

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, test_imgs, train_event_stack, test_event_stack, train_bb, test_proposals, ):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        num_sequences = train_imgs.shape[-4]
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1
        num_test_images = test_imgs.shape[0] if test_imgs.dim() == 5 else 1
        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        train_event_feat1, train_event_feat2 = self.motion_attention(train_event_stack[0], train_event_stack[1], train_event_stack[2])
        test_event_feat1, test_event_feat2 = self.motion_attention(test_event_stack[0], test_event_stack[1], test_event_stack[2])

        [x1, x2] = self.counter_guide(train_feat['layer2'], train_feat['layer3'], train_event_feat1, train_event_feat2)
        train_feat['layer2'] = x1
        train_feat['layer3'] = x2

        [x3, x4] = self.counter_guide(test_feat['layer2'], test_feat['layer3'], test_event_feat1, test_event_feat2)
        test_feat['layer2'] = x3
        test_feat['layer3'] = x4

        # extrack event features -------------> for channel+spatial attention
        # train_event_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_event_stack.shape[-3:]))
        # test_event_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_event_stack.shape[-3:]))
        # [x1, x2] = self.inter_atten1([train_feat['layer2'], train_feat['layer3']], \
        #                              [train_event_feat['layer2'], train_event_feat['layer3']])
        # train_feat['layer2'] = x1
        # train_feat['layer3'] = x2
        # [x3, x4] = self.inter_atten2([test_feat['layer2'], test_feat['layer3']], \
        #                             [test_event_feat['layer2'], test_event_feat['layer3']])
        # test_feat['layer2'] = x3
        # test_feat['layer3'] = x4



        train_feat_iou = [feat for feat in train_feat.values()]
        test_feat_iou = [feat for feat in test_feat.values()]

        # Obtain iou prediction
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou,
                                     train_bb.reshape(num_train_images, num_sequences, 4),
                                     test_proposals.reshape(num_train_images, num_sequences, -1, 4))
        return iou_pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)



@model_constructor
def atom_resnet18(iou_input_dim=(256,256), iou_inter_dim=(256,256), backbone_pretrained=True):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    # Bounding box regressor
    iou_predictor = bbmodels.AtomIoUNet(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    net = ATOMnet(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                  extractor_grad=False)
    # for name, parms in net.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad)

    return net


@model_constructor
def atom_resnet50(iou_input_dim=(256,256), iou_inter_dim=(256,256), backbone_pretrained=True):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # Bounding box regressor
    iou_predictor = bbmodels.AtomIoUNet(input_dim=(4*128,4*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    net = ATOMnet(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                  extractor_grad=False)

    return net

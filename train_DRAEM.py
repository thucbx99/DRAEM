import os

import cv2
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data_loader import MVTecDRAEMTrainDataset, MVTecDRAEMTestDataset
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
from logger import CompleteLogger
from meter import AverageMeter, ProgressMeter


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def test(obj_name, model, model_seg, args):
    # switch to eval mode
    model.eval()
    model_seg.eval()

    dataset = MVTecDRAEMTestDataset(os.path.join(args.data_path, obj_name, 'test'), resize_shape=[256, 256])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    total_pixel_scores = np.zeros((256 * 256 * len(dataset)))
    total_gt_pixel_scores = np.zeros((256 * 256 * len(dataset)))
    mask_cnt = 0
    anomaly_score_gt = []
    anomaly_score_prediction = []

    for i_batch, sample_batched in enumerate(dataloader):
        gray_batch = sample_batched["image"].cuda()

        is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
        anomaly_score_gt.append(is_normal)
        true_mask = sample_batched["mask"]
        true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

        gray_rec = model(gray_batch)
        joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

        out_mask = model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)

        out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()
        out_mask_averaged = F.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1,
                                         padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)

        anomaly_score_prediction.append(image_score)

        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        total_pixel_scores[mask_cnt * 256 * 256:(mask_cnt + 1) * 256 * 256] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * 256 * 256:(mask_cnt + 1) * 256 * 256] = flat_true_mask
        mask_cnt += 1

    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)
    auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
    ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:256 * 256 * mask_cnt]
    total_pixel_scores = total_pixel_scores[:256 * 256 * mask_cnt]
    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)

    print("=" * 50)
    print("AUC Image:  " + str(auroc))
    print("AP Image:  " + str(ap))
    print("AUC Pixel:  " + str(auroc_pixel))
    print("AP Pixel:  " + str(ap_pixel))
    print("=" * 50)

    return ap_pixel


def train_on_device(obj_names, args):
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    logger = CompleteLogger(args.log_path)

    def visualize(batch_image, name):
        """
        Args:
            batch_image (tensor): N x 3 x H x W
            name: filename of the saving image
        """
        batch_image = batch_image.detach().cpu().numpy()
        for idx, image in enumerate(batch_image):
            image = image.transpose((1, 2, 0)) * 255
            image = Image.fromarray(np.uint8(image))
            image.save(logger.get_image_path("{}_{}.png".format(name, idx)))

    def visualize_mask(batch_mask, name):
        batch_mask = batch_mask.detach().cpu().numpy()
        for idx, mask in enumerate(batch_mask):
            mask = mask * 255
            cv2.imwrite(logger.get_image_path("{}_{}.png".format(name, idx)), mask)

    print(obj_names)
    for obj_name in obj_names:
        print('anomaly detection object {}'.format(obj_name))

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.cuda()
        model.apply(weights_init)

        if args.pretrained_generative:
            model.load_state_dict(torch.load(args.pretrained_generative, map_location='cpu'))

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.cuda()
        model_seg.apply(weights_init)

        if args.pretrained_discriminative:
            model_seg.load_state_dict(torch.load(args.pretrained_discriminative, map_location='cpu'))

        optimizer = torch.optim.Adam([
            {"params": model.parameters(), "lr": args.lr},
            {"params": model_seg.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs * 0.8, args.epochs * 0.9], gamma=0.2,
                                                   last_epoch=-1)

        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()

        dataset = MVTecDRAEMTrainDataset(os.path.join(args.data_path, obj_name, 'train', 'good'),
                                         args.anomaly_source_path, resize_shape=[256, 256])
        subset_size = int(len(dataset) * args.sample_rate)
        subset_idxes = np.random.choice(np.arange(len(dataset)), subset_size, replace=False)
        dataset = Subset(dataset, subset_idxes)
        print('dataset_size: {}'.format(len(dataset)))

        dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=16)

        best_ap = 0
        for epoch in range(args.epochs):
            logger.set_epoch(epoch)
            model.train()
            model_seg.train()

            l2_losses = AverageMeter('L2 Loss', ':3.2f')
            ssim_losses = AverageMeter('SSIM Loss', ':3.2f')
            segment_losses = AverageMeter('Segment Loss', ':3.2f')
            losses = AverageMeter('Loss', ':3.2f')

            progress = ProgressMeter(
                len(dataloader),
                [l2_losses, ssim_losses, segment_losses, losses],
                prefix="Epoch: [{}]".format(epoch))

            # train for one epoch
            print("Epoch: " + str(epoch))
            for i_batch, sample_batched in enumerate(dataloader):
                gray_batch = sample_batched["image"].cuda()
                aug_gray_batch = sample_batched["augmented_image"].cuda()
                anomaly_mask = sample_batched["anomaly_mask"].cuda()

                gray_rec = model(aug_gray_batch)
                joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                l2_loss = loss_l2(gray_rec, gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)
                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                loss = l2_loss + ssim_loss + segment_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                l2_losses.update(l2_loss.item(), args.bs)
                ssim_losses.update(ssim_loss.item(), args.bs)
                segment_losses.update(segment_loss.item(), args.bs)
                losses.update(loss.item(), args.bs)

                if i_batch % args.print_freq == 0:
                    progress.display(i_batch)
                    visualize(gray_rec, 'rec')
                    visualize(gray_batch, 'origin')
                    visualize(aug_gray_batch, 'augmented')

                    visualize_mask(out_mask_sm[:, 1, :, :], 'output_mask')
                    visualize_mask(anomaly_mask.squeeze(1), 'gt_mask')

            scheduler.step()

            # evaluate
            ap = test(obj_name, model, model_seg, args)
            best_ap = max(ap, best_ap)

            # save checkpoints

            torch.save(model.state_dict(), logger.get_checkpoint_path('latest_generative'))
            torch.save(model_seg.state_dict(), logger.get_checkpoint_path('latest_discriminative'))

    print('best AP {}'.format(best_ap))
    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--sample_rate', default=0.5, type=float,
                        help='sampling rate for transfer learning')
    parser.add_argument('--print_freq', default=5, type=int)
    parser.add_argument('--pretrained-generative', default=None, type=str)
    parser.add_argument('--pretrained-discriminative', default=None, type=str)

    args = parser.parse_args()

    obj_batch = [['capsule'],
                 ['bottle'],
                 ['carpet'],
                 ['leather'],
                 ['pill'],
                 ['transistor'],
                 ['tile'],
                 ['cable'],
                 ['zipper'],
                 ['toothbrush'],
                 ['metal_nut'],
                 ['hazelnut'],
                 ['screw'],
                 ['grid'],
                 ['wood']
                 ]

    if int(args.obj_id) == -1:
        obj_list = ['capsule',
                    'bottle',
                    'carpet',
                    'leather',
                    'pill',
                    'transistor',
                    'tile',
                    'cable',
                    'zipper',
                    'toothbrush',
                    'metal_nut',
                    'hazelnut',
                    'screw',
                    'grid',
                    'wood'
                    ]
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    with torch.cuda.device(args.gpu_id):
        train_on_device(picked_classes, args)

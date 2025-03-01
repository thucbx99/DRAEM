import os
import shutil
from itertools import cycle

import cv2
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from data_loader import MVTecDRAEMTrainDataset, MVTecDRAEMTestDataset
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
from logger import CompleteLogger
from meter import AverageMeter, ProgressMeter


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def test(obj_name, model, model_seg, args):
    print('testing on object {}'.format(obj_name))
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


def train_on_device(args):
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

    # joint training on carpet and grid
    model_carpet = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model_grid = ReconstructiveSubNetwork(in_channels=3, out_channels=3)

    model_carpet.cuda()
    model_carpet.apply(weights_init)

    model_grid.cuda()
    model_grid.apply(weights_init)

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.cuda()
    model_seg.apply(weights_init)

    optimizer = torch.optim.Adam([
        {"params": model_carpet.parameters(), "lr": args.lr},
        {"params": model_grid.parameters(), "lr": args.lr},
        {"params": model_seg.parameters(), "lr": args.lr}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs * 0.8, args.epochs * 0.9], gamma=0.2,
                                               last_epoch=-1)

    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    loss_focal = FocalLoss()

    carpet_dataset = MVTecDRAEMTrainDataset(os.path.join(args.data_path, 'carpet', 'train', 'good'),
                                            args.anomaly_source_path, resize_shape=[256, 256])
    grid_dataset = MVTecDRAEMTrainDataset(os.path.join(args.data_path, 'grid', 'train', 'good'),
                                          args.anomaly_source_path, resize_shape=[256, 256])

    carpet_loader = DataLoader(carpet_dataset, batch_size=args.bs // 2, shuffle=True, num_workers=4)
    grid_loader = DataLoader(grid_dataset, batch_size=args.bs // 2, shuffle=True, num_workers=4)

    best_ap = 0
    corr_carpet_ap = 0
    corr_gird_ap = 0

    for epoch in range(args.epochs):
        logger.set_epoch(epoch)
        model_carpet.train()
        model_grid.train()
        model_seg.train()

        l2_losses = AverageMeter('L2 Loss', ':3.2f')
        ssim_losses = AverageMeter('SSIM Loss', ':3.2f')
        segment_losses = AverageMeter('Segment Loss', ':3.2f')
        losses = AverageMeter('Loss', ':3.2f')

        progress = ProgressMeter(
            len(carpet_loader),
            [l2_losses, ssim_losses, segment_losses, losses],
            prefix="Epoch: [{}]".format(epoch))

        # train for one epoch
        print("Epoch: " + str(epoch))
        for idx, (carpet_batch, grid_batch) in enumerate(zip(carpet_loader, cycle(grid_loader))):

            # carpet data
            x_carpet = carpet_batch["image"].cuda()
            x_aug_carpet = carpet_batch["augmented_image"].cuda()
            mask_carpet = carpet_batch["anomaly_mask"].cuda()

            # grid data
            x_grid = grid_batch["image"].cuda()
            x_aug_grid = grid_batch["augmented_image"].cuda()
            mask_grid = grid_batch["anomaly_mask"].cuda()

            # compute output
            carpet_rec = model_carpet(x_aug_carpet)
            carpet_join = torch.cat((carpet_rec, x_aug_carpet), dim=1)

            grid_rec = model_grid(x_aug_grid)
            grid_join = torch.cat((grid_rec, x_aug_grid), dim=1)

            out_mask_carpet = model_seg(carpet_join)
            out_mask_carpet = torch.softmax(out_mask_carpet, dim=1)
            out_mask_grid = model_seg(grid_join)
            out_mask_grid = torch.softmax(out_mask_grid, dim=1)

            l2_loss = (loss_l2(carpet_rec, x_carpet) + loss_l2(grid_rec, x_grid)) / 2
            ssim_loss = (loss_ssim(carpet_rec, x_carpet) + loss_ssim(grid_rec, x_grid)) / 2
            segment_loss = (loss_focal(out_mask_carpet, mask_carpet) + loss_focal(out_mask_grid, mask_grid)) / 2
            loss = l2_loss + ssim_loss + segment_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            l2_losses.update(l2_loss.item(), args.bs)
            ssim_losses.update(ssim_loss.item(), args.bs)
            segment_losses.update(segment_loss.item(), args.bs)
            losses.update(loss.item(), args.bs)

            if idx % args.print_freq == 0:
                progress.display(idx)

                visualize(carpet_rec, 'carpet_rec')
                visualize(x_carpet, 'x_carpet')
                visualize(x_aug_carpet, 'x_aug_carpet')
                visualize_mask(mask_carpet.squeeze(1), 'carpet_gt')
                visualize_mask(out_mask_carpet[:, 1, :, :], 'carpet_output')

                visualize(grid_rec, 'grid_rec')
                visualize(x_grid, 'x_grid')
                visualize(x_aug_grid, 'x_aug_grid')
                visualize_mask(mask_grid.squeeze(1), 'grid_gt')
                visualize_mask(out_mask_grid[:, 1, :, :], 'grid_output')

        scheduler.step()

        # evaluate
        carpet_ap = test('carpet', model_carpet, model_seg, args)
        grid_ap = test('grid', model_grid, model_seg, args)

        # save checkpoints
        torch.save(model_carpet.state_dict(), logger.get_checkpoint_path('latest_carpet'))
        torch.save(model_grid.state_dict(), logger.get_checkpoint_path('latest_grid'))
        torch.save(model_seg.state_dict(), logger.get_checkpoint_path('latest_discriminative'))

        if carpet_ap + grid_ap > best_ap:
            best_ap = carpet_ap + grid_ap
            corr_carpet_ap = carpet_ap
            corr_gird_ap = grid_ap
            shutil.copy(logger.get_checkpoint_path('latest_carpet'), logger.get_checkpoint_path('best_carpet'))
            shutil.copy(logger.get_checkpoint_path('latest_grid'), logger.get_checkpoint_path('best_grid'))
            shutil.copy(logger.get_checkpoint_path('latest_discriminative'),
                        logger.get_checkpoint_path('best_discriminative'))

    print('performance of the best checkpoint carpet {}, grid {}'.format(corr_carpet_ap, corr_gird_ap))
    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--print_freq', default=5, type=int)

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

    with torch.cuda.device(args.gpu_id):
        train_on_device(args)

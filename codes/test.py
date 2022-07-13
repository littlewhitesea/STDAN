'''
test STDAN models on arbitrary datasets
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import time

import utils.util as util
import data.util as data_util
import models.modules.sttrans as sttrans

def main():
    scale = 4
    N_ot = 7 #3
    N_in = 1+ N_ot // 2

    torch.backends.cudnn.benchmark = True

    #### model 
    #### TODO: change your model path herec
    model_path = '/media/DATA2/wzr/ECCV_comparisons/github_upload/STDAN/experiments/pretrained_model/STDAN_20220306.pth'
    model = sttrans.STTrans2(scale=4, n_inputs=4, n_outputs=7, nf=64, embed_dim=72, window_size=8)

    #### dataset
    data_mode = 'Custom' #'Vid4' #'SPMC'#'Middlebury'#

    if data_mode == 'Vid4':
        test_dataset_folder = '/media/DATA2/wzr/Vid4+UDM+SPMC-11/Vid4/lr_x4/*'
        # TODO: modify the path according to your data
    if data_mode == 'SPMC':
        test_dataset_folder = '/media/data3/wh/data/dataset/SR/SPMC-11/*'
        # TODO: modify the path according to your data
    if data_mode == 'Custom':
        test_dataset_folder = '../test_example/*' # TODO: put your own data path here

    #### evaluation
    flip_test = False #True#
    crop_border = 0

    # temporal padding mode
    padding = 'zero padding'
    save_imgs = True #True#
    if 'Custom' in data_mode: save_imgs = True
    ############################################################################
    if torch.cuda.is_available():
        device = torch.device('cuda') 
    else:
        device = torch.device('cpu')
    save_folder = '../results/{}'.format(data_mode)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    model_params = util.get_model_total_params(model)

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Model parameters: {} M'.format(model_params))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip Test: {}'.format(flip_test))
   

    def single_forward(model, imgs_in):
        with torch.no_grad():
            b,n,c,h,w = imgs_in.size()
            h_n = int(8*np.ceil(h/8))
            w_n = int(8*np.ceil(w/8))
            imgs_temp = imgs_in.new_zeros(b,n,c,h_n,w_n)
            imgs_temp[:,:,:,0:h,0:w] = imgs_in

            start = time.time()
            model_output = model(imgs_temp)
            end = time.time()
            print(end - start)

            model_output = model_output[:, :, :, 0:scale*h, 0:scale*w]
            if isinstance(model_output, list) or isinstance(model_output, tuple):
                output = model_output[0]
            else:
                output = model_output
        return output

    sub_folder_l = sorted(glob.glob(test_dataset_folder))
    model.load_state_dict(torch.load(model_path), strict=True)

    model.eval()
    model = model.to(device)

    avg_psnr_l = []
    avg_psnr_y_l = []
    sub_folder_name_l = []

    avg_ssim_l = []
    avg_ssim_y_l = []

    for sub_folder in sub_folder_l:
        gt_tested_list = []
        sub_folder_name = sub_folder.split('/')[-1]
        sub_folder_name_l.append(sub_folder_name)
        save_sub_folder = osp.join(save_folder, sub_folder_name)

        if data_mode == 'SPMC':
            sub_folder = sub_folder + '/LR/'
        img_LR_l = sorted(glob.glob(sub_folder + '/*'))

        if save_imgs:
            util.mkdirs(save_sub_folder)

        #### read LR images
        imgs = util.read_seq_imgs(sub_folder)
        #### read GT images
        img_GT_l = []
        if data_mode == 'SPMC':
            sub_folder_GT = osp.join(sub_folder.replace('/LR/', '/truth/'))
        else:
            sub_folder_GT = osp.join(sub_folder.replace('/lr_x4/', '/hr/'))

        if 'Custom' not in data_mode:
            for img_GT_path in sorted(glob.glob(osp.join(sub_folder_GT,'*'))):
                img_GT_l.append(util.read_image(img_GT_path))

        avg_psnr, avg_psnr_sum, cal_n = 0,0,0
        avg_psnr_y, avg_psnr_sum_y = 0,0

        avg_ssim, avg_ssim_sum = 0, 0
        avg_ssim_y, avg_ssim_sum_y = 0, 0
        
        if len(img_LR_l) == len(img_GT_l):
            skip = True
        else:
            skip = False
        
        if 'Custom' in data_mode:
            select_idx_list = util.test_index_generation(False, N_ot, len(img_LR_l))
        else:
            select_idx_list = util.test_index_generation(skip, N_ot, len(img_LR_l))
        # process each image
        for select_idxs in select_idx_list:
            # get input images
            select_idx = select_idxs[0]
            gt_idx = select_idxs[1]
            imgs_in = imgs.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

            output = single_forward(model, imgs_in)

            outputs = output.data.float().cpu().squeeze(0)            

            if flip_test:
                # flip W
                output = single_forward(model, torch.flip(imgs_in, (-1, )))
                output = torch.flip(output, (-1, ))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output
                # flip H
                output = single_forward(model, torch.flip(imgs_in, (-2, )))
                output = torch.flip(output, (-2, ))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output
                # flip both H and W
                output = single_forward(model, torch.flip(imgs_in, (-2, -1)))
                output = torch.flip(output, (-2, -1))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output

                outputs = outputs / 4

            # save imgs
            for idx, name_idx in enumerate(gt_idx):
                if name_idx in gt_tested_list:
                    continue
                gt_tested_list.append(name_idx)
                output_f = outputs[idx,:,:,:].squeeze(0)

                output = util.tensor2img(output_f)
                if save_imgs:                
                    cv2.imwrite(osp.join(save_sub_folder, '{:08d}.png'.format(name_idx+1)), output)

                if 'Custom' not in data_mode:
                    #### calculate PSNR
                    output = output / 255.

                    GT = np.copy(img_GT_l[name_idx])

                    if crop_border == 0:
                        cropped_output = output
                        cropped_GT = GT
                    else:
                        cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
                        cropped_GT = GT[crop_border:-crop_border, crop_border:-crop_border, :]
                    crt_psnr = util.calculate_psnr(cropped_output * 255, cropped_GT * 255)
                    cropped_GT_y = data_util.bgr2ycbcr(cropped_GT, only_y=True)
                    cropped_output_y = data_util.bgr2ycbcr(cropped_output, only_y=True)
                    crt_psnr_y = util.calculate_psnr(cropped_output_y * 255, cropped_GT_y * 255)
                    logger.info('{:3d} - {:25}.png \tPSNR: {:.6f} dB  PSNR-Y: {:.6f} dB'.format(name_idx + 1, name_idx+1, crt_psnr, crt_psnr_y))
                    avg_psnr_sum += crt_psnr
                    avg_psnr_sum_y += crt_psnr_y

                    crt_ssim = util.calculate_ssim(cropped_output, cropped_GT)
                    crt_ssim_y = util.calculate_ssim(cropped_output_y * 255, cropped_GT_y * 255)
                    logger.info('{:3d} - {:25}.png \tSSIM: {:.6f}  SSIM-Y: {:.6f}'.format(name_idx + 1, name_idx+1, crt_ssim, crt_ssim_y))
                    avg_ssim_sum += crt_ssim
                    avg_ssim_sum_y += crt_ssim_y


                    cal_n += 1

        if 'Custom' not in data_mode:
            avg_psnr = avg_psnr_sum / cal_n
            avg_psnr_y = avg_psnr_sum_y / cal_n
    
            logger.info('Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB for {} frames; '.format(sub_folder_name, avg_psnr, avg_psnr_y, cal_n))
    
            avg_psnr_l.append(avg_psnr)
            avg_psnr_y_l.append(avg_psnr_y)

            # modified here
            avg_ssim = avg_ssim_sum / cal_n
            avg_ssim_y = avg_ssim_sum_y / cal_n

            logger.info('Folder {} - Average SSIM: {:.6f} SSIM-Y: {:.6f} for {} frames; '.format(sub_folder_name,
                                                                                                       avg_ssim,
                                                                                                       avg_ssim_y,
                                                                                                       cal_n))

            avg_ssim_l.append(avg_ssim)
            avg_ssim_y_l.append(avg_ssim_y)


    if 'Custom' not in data_mode:
        logger.info('################ Tidy Outputs ################')
        for name, psnr, psnr_y in zip(sub_folder_name_l, avg_psnr_l, avg_psnr_y_l):
            logger.info('Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB. '
                       .format(name, psnr, psnr_y))
        #     modified here
        for name, ssim, ssim_y in zip(sub_folder_name_l, avg_ssim_l, avg_ssim_y_l):
            logger.info('Folder {} - Average SSIM: {:.6f} SSIM-Y: {:.6f}. '
                       .format(name, ssim, ssim_y))
        logger.info('################ Final Results ################')
        logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
        logger.info('Padding mode: {}'.format(padding))
        logger.info('Model path: {}'.format(model_path))
        logger.info('Save images: {}'.format(save_imgs))
        logger.info('Flip Test: {}'.format(flip_test))
        logger.info('Total Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB for {} clips. '
                    .format(
                        sum(avg_psnr_l) / len(avg_psnr_l), sum(avg_psnr_y_l) / len(avg_psnr_y_l), len(sub_folder_l)))

        logger.info('Total Average SSIM: {:.6f} SSIM-Y: {:.6f} for {} clips. '
                    .format(
                        sum(avg_ssim_l) / len(avg_ssim_l), sum(avg_ssim_y_l) / len(avg_ssim_y_l), len(sub_folder_l)))

if __name__ == '__main__':
    main()

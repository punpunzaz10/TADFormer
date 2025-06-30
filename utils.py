# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

# # This code is referenced from
# Repository: https://github.com/scale-lab/MTLoRA



import os
import torch
import torch.distributed as dist
from torch import inf
import errno

from PIL import Image
import numpy as np
import cv2
import imageio
import scipy.io as sio
import torch.nn.functional as F
from models.lora import map_old_state_dict_weights



def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger, backbone=False, quiet=False, bestmodel=False, bestepoch=None):
    resume_path = config.MODEL.RESUME if not backbone else config.MODEL.RESUME_BACKBONE

    if bestmodel:
        resume_path = os.path.join(config.OUTPUT, f"bestmodel_epoch_{bestepoch}.pth")

    logger.info(
        f"==============> Resuming form {resume_path}....................")
    if resume_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            resume_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(resume_path, map_location='cpu')

    tadmtl = config.MODEL.TADMTL
    tadmtl_enabled = tadmtl.ENABLED

    skip_decoder = config.TRAIN.SKIP_DECODER_CKPT

    model_state = {k: v for k, v in checkpoint["model"].items(
    ) if not k.startswith("decoders")} if skip_decoder else checkpoint["model"]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in model_state.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del model_state[k]

    if config.MODEL.UPDATE_RELATIVE_POSITION:
        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [
            k for k in model_state.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del model_state[k]

        # delete relative_coords_table since we always re-init it
        relative_position_index_keys = [
            k for k in model_state.keys() if "relative_coords_table" in k]
        for k in relative_position_index_keys:
            del model_state[k]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [
            k for k in model_state.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = model_state[k]
            relative_position_bias_table_current = model.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    # bicubic interpolate relative_position_bias_table if not match
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                        mode='bicubic')
                    model_state[k] = relative_position_bias_table_pretrained_resized.view(
                        nH2, L2).permute(1, 0)

        # bicubic interpolate absolute_pos_embed if not match
        absolute_pos_embed_keys = [
            k for k in model_state.keys() if "absolute_pos_embed" in k]
        for k in absolute_pos_embed_keys:
            # dpe
            absolute_pos_embed_pretrained = model_state[k]
            absolute_pos_embed_current = model.model_state()[k]
            _, L1, C1 = absolute_pos_embed_pretrained.size()
            _, L2, C2 = absolute_pos_embed_current.size()
            if C1 != C1:
                logger.warning(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(
                        -1, S1, S1, C1)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(
                        0, 3, 1, 2)
                    absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                        absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(
                        0, 2, 3, 1)
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(
                        1, 2)
                    model_state[k] = absolute_pos_embed_pretrained_resized

    if tadmtl_enabled:
        mapping = {}
        trainable_layers = []
        if tadmtl.QKV_ENABLED:
            trainable_layers.extend(["attn.qkv.weight", "attn.qkv.bias"])
        if tadmtl.PROJ_ENABLED:
            trainable_layers.extend(["attn.proj.weight", "attn.proj.bias"])
        if tadmtl.FC1_ENABLED:
            trainable_layers.extend(["mlp.fc1.weight", "mlp.fc1.bias"])
        if tadmtl.FC2_ENABLED:
            trainable_layers.extend(["mlp.fc2.weight", "mlp.fc2.bias"])
        if tadmtl.DOWNSAMPLER_ENABLED:
            trainable_layers.extend(["downsample.reduction.weight"])

        for k, v in model_state.items():
            last_three = ".".join(k.split(".")[-3:])
            prefix = ".".join(k.split(".")[:-3])
            if last_three in trainable_layers:
                weight_bias = last_three.split(".")[-1]
                layer_name = ".".join(last_three.split(".")[:-1])
                mapping[f"{prefix}.{layer_name}.{weight_bias}"] = f"{prefix}.{layer_name}.linear.{weight_bias}"
        if not len(mapping):
            print("No keys needs to be mapped for LoRA")
        model_state = map_old_state_dict_weights(
            model_state, mapping, "", config.MODEL.TADMTL.SPLIT_QKV)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if not quiet:
        if len(missing) > 0:
            logger.warning("=============Missing Keys==============")
            for k in missing:
                logger.warning(k)
        if len(unexpected) > 0:
            logger.warning("=============Unexpected Keys==============")
            for k in unexpected:
                logger.warning(k)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint and not skip_decoder:
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(
            f"=> loaded successfully '{resume_path}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy





def load_pretrained(config, model, logger):
    logger.info(
        f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [
        k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [
        k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(
                    nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [
        k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(
                    -1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(
                    0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(
                    0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(
                    1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(
                f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    save_name = f'ckpt_epoch_{epoch}.pth'
    save_path = os.path.join(config.OUTPUT, save_name)
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    return save_path

def save_best_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    save_name = f'bestmodel_epoch_{epoch}.pth'
    save_path = os.path.join(config.OUTPUT, save_name)
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    return save_path


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d)
                                for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device)
                         for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):

        self._scaler.scale(loss).backward(create_graph=create_graph)

        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                # unscale the gradients of optimizer's assigned params in-place
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm


    def compute_task_gradients(self, losses, optimizer):
        """
        Compute gradients for multiple tasks and return their cosine similarities.
        :param losses: List of losses for different tasks.
        :param optimizer: Optimizer used for training.
        :return: List of cosine similarities between the gradients of consecutive tasks.
        """
        gradients = []
        for loss in losses:
            optimizer.zero_grad()  # Clear previous gradients
            self._scaler.scale(loss).backward()  # Compute gradient for the current task
            grads = []
            for param in optimizer.param_groups[0]['params']:
                if param.grad is not None:
                    grads.append(param.grad.detach().clone())
            gradients.append(grads)

        # Compute cosine similarities between consecutive tasks
        cosine_similarities = []
        for i in range(len(gradients) - 1):
            flat_grads_i = torch.cat([g.view(-1) for g in gradients[i]])
            flat_grads_j = torch.cat([g.view(-1) for g in gradients[i + 1]])
            cosine_similarity = torch.nn.functional.cosine_similarity(flat_grads_i, flat_grads_j, dim=0)
            cosine_similarities.append(cosine_similarity.item())

        return cosine_similarities


    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def tens2image(tens, transpose=False):
    """Converts tensor with 2 or 3 dimensions to numpy array"""
    im = tens.cpu().detach().numpy()

    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)
    elif im.shape[-1] == 1:
        im = np.squeeze(im)
    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)
    if transpose:
        if im.ndim == 3:
            im = im.transpose((1, 2, 0))
    return im


def get_output(output, task):

    if task == 'normals':
        output = output.permute(0, 2, 3, 1)
        output = (F.normalize(output, p=2, dim=3) + 1.0) * 255 / 2.0

    elif task in {'semseg'}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)


    elif task in {'human_parts'}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)

    elif task in {'edge'}:
        output = output.permute(0, 2, 3, 1)
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)), dim=3)

    elif task in {'sal'}:
        output = output.permute(0, 2, 3, 1)

        #output[sal].shape: torch.Size([1, 1, 448, 448]) -> [1, 448, 448, 1]

        #output = F.softmax(output, dim=3)[:, :, :, 1] * 255  # torch.squeeze(255 * 1 / (1 + torch.exp(-output)))

        # My Revised
        output = F.softmax(output, dim=3)[:, :, :, 0] * 255  # torch.squeeze(255 * 1 / (1 + torch.exp(-output)))

    elif task in {'depth'}:
        output.clamp_(min=0.)
        output = output.permute(0, 2, 3, 1)

    else:
        raise ValueError('Select one of the valid tasks')

    return output



def normalize(arr, t_min=0, t_max=255):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = arr.max() - arr.min()
    for i in arr:
        temp = (((i - arr.min())*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    res = np.array(norm_arr)
    return res



def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])



def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype = np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ ( np.uint8(str_id[-1]) << (7-j))
            g = g ^ ( np.uint8(str_id[-2]) << (7-j))
            b = b ^ ( np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap



def vis_semseg(_semseg):

    new_cmap = labelcolormap(21)
    _semseg = new_cmap[_semseg]
    return _semseg


def vis_parts(inp):
    new_cmap = labelcolormap(7)
    inp = new_cmap[inp]
    return inp



def save_model_pred_for_one_task(p, sample, output, save_dirs, task=None, epoch=None):
    """ Save model predictions for one task"""

    inputs, meta = sample['image'].cuda(non_blocking=True), sample['meta']
    ignore_index = 255

    if task == 'semseg':
        if not p.semseg_save_train_class and p.train_db_name == 'Cityscapes3D':
            output_task = get_output(output[task], task, semseg_save_train_class=False).cpu().data.numpy()
        else:
            output_task = get_output(output[task], task).cpu().data.numpy()

    elif task == '3ddet': # save only the first iteraction in an epoch for examing the performance
        from detection_toolbox.det_tools import bbox2json, bbox2fig
        det_res_list = get_output(output[task], task, p=p, label=sample)
        bs = int(inputs.size()[0])
        K_matrixes = sample['meta']['K_matrix'].cpu().numpy()
        cam_params = [{k: v[sa] for k, v in sample['bbox_camera_params'].items()} for sa in range(bs)]

        if batch_idx == 0:
            # get gt labels
            gt_center_I = []
            gt_center_S = []
            gt_size_S = []
            gt_rotation_S = []
            gt_class = []
            for _i in range(bs):
                if type(sample['det_labels'][_i]) == dict:
                    gt_center_I.append(sample['det_labels'][_i]['center_I'].cpu().numpy())
                    gt_center_S.append(sample['det_labels'][_i]['center_S'].cpu().numpy())
                    gt_size_S.append(sample['det_labels'][_i]['size_S'].cpu().numpy())
                    gt_rotation_S.append(sample['det_labels'][_i]['rotation_S'].cpu().numpy())
                    gt_class.append(sample['det_labels'][_i]['label'])
                else:
                    gt_center_I.append(None)
                    gt_center_S.append(None)
                    gt_size_S.append(None)
                    gt_rotation_S.append(None)
                    gt_class.append(None)

        for jj in range(bs):
            fname = meta['img_name'][jj]
            vis_fname = 'it' + str(epoch) + '_' + meta['img_name'][jj]
            # save bbox predictions in cityscapes evaluation format
            json_dict = bbox2json(det_res_list[jj], K_matrixes[jj], cam_params[jj])
            out_path = os.path.join(save_dirs[task], fname + '.json')
            with open(out_path, 'w') as outfile:
                json.dump(json_dict, outfile)
            if True and batch_idx ==0:
                # visualization, but it takes time so we only use it in infer mode
                box_no = len(det_res_list[jj]['img_bbox']['scores_3d'])
                if box_no > 0:
                    gt_labels = [gt_class[jj], gt_center_I[jj], gt_center_S[jj], gt_size_S[jj], gt_rotation_S[jj]]
                    vis_fig = bbox2fig(p, inputs[jj].cpu(), det_res_list[jj], K_matrixes[jj], cam_params[jj], gt_labels)
                    imageio.imwrite(os.path.join(save_dirs[task], vis_fname + '_' + str(box_no) + '.png'), vis_fig.astype(np.uint8))

        return
    else:
        output_task = get_output(output[task], task)#.cpu().data.numpy()

    for jj in range(int(inputs.size()[0])):
        if len(sample[task][jj].unique()) == 1 and sample[task][jj].unique() == ignore_index:
            continue
        fname = meta['image'][jj]

        im_height = meta['im_size'][0][jj]
        im_width = meta['im_size'][1][jj]

        if im_width == 500:
            pass

        pred = output_task[jj] # (H, W) or (H, W, C)
        # if we used padding on the input, we crop the prediction accordingly
        # if (im_height, im_width) != pred.shape[:2]:
        #     delta_height = max(pred.shape[0] - im_height, 0)
        #     delta_width = max(pred.shape[1] - im_width, 0)
        #     if delta_height > 0 or delta_width > 0:
        #         height_begin = torch.div(delta_height, 2, rounding_mode="trunc")
        #         height_location = [height_begin, height_begin + im_height]
        #         width_begin =torch.div(delta_width, 2, rounding_mode="trunc")
        #         width_location = [width_begin, width_begin + im_width]
        #         pred = pred[height_location[0]:height_location[1],
        #                     width_location[0]:width_location[1]]
            # if pred.shape[1] < im_width:
            #     # 필요한 만큼 좌우에 (im_width - pred_w) / 2 씩 패딩
            #     pad_total = im_width - pred.shape[1]
            #     left_pad = pad_total // 2
            #     right_pad = pad_total - left_pad
            #     # 만약 pred가 [H, W]라면
            #     pred = F.pad(pred, (left_pad, right_pad), mode='constant', value=0)

        #assert pred.shape[:2] == (im_height, im_width)
        if pred.ndim == 3:
            raise
        result = pred.cpu().numpy()
        if task == 'depth':
            sio.savemat(os.path.join(save_dirs[task], fname + '.mat'), {'depth': result})
        else:
            imageio.imwrite(os.path.join(save_dirs, fname + '.png'), result.astype(np.uint8))



def save_imgs_mtl(batch_imgs, batch_labels, path, id):

    # 수정할 부분 : 1) normalized gt, sal output,
    import torchvision

    imgs = tens2image(batch_imgs, transpose=True)
    labels = {task: tens2image(label, transpose=True)
              for task, label in batch_labels.items()}
    #predictions = {task: tens2image(prediction)
                   #for task, prediction in batch_predictions.items()}

    # Image.fromarray(normalize(imgs, 0, 255).astype(
    #     np.uint8)).save(f'{path}/{id}_img.png')

    task='edge'

    labels[task] = normalize(labels[task], 0, 255)
    #predictions[task] = normalize(predictions[task], 0, 255)
    Image.fromarray(labels[task].astype(np.uint8)).save(
        f'{path}/{id}.png')
    # Image.fromarray(predictions[task].astype(np.uint8)).save(
    #     f'{path}/{id}_{task}_pred.png')


    # for task in labels.keys():
    #     if task == "semseg":
    #         print(np.sum(labels[task] != 255))
    #         labels[task] = labels[task] != 255
    #         predictions[task] = predictions[task] != 225
    #         batch_imgs = 255*(batch_imgs-torch.min(batch_imgs)) / \
    #             (torch.max(batch_imgs)-torch.min(batch_imgs))
    #
    #         batch_predictions[task][0][batch_predictions[task][0] == 255] = 0
    #         vis_seg = vis_semseg(batch_predictions[task][0].cpu())
    #         Image.fromarray(vis_seg).save(f'{path}/{id}_{task}_pred.png')
    #         seg_gt = batch_labels[task][:, 0, :, :]
    #         seg_gt[0][seg_gt[0] == 255] = 0
    #         vis_seg_gt = vis_semseg(seg_gt[0].cpu().detach().to(torch.uint8))
    #         Image.fromarray(vis_seg_gt).save(f'{path}/{id}_{task}_gt.png')
    #
    #         semseg = torchvision.utils.draw_segmentation_masks(batch_imgs[0].cpu().detach().to(torch.uint8),
    #                                                            batch_predictions[task][0].to(torch.bool), colors="blue", alpha=0.5)
    #         Image.fromarray(semseg.numpy().transpose((1, 2, 0))
    #                         ).save(f'{path}/{id}_{task}_pred_masked.png')
    #         semseg = torchvision.utils.draw_segmentation_masks(batch_imgs[0].cpu().detach().to(torch.uint8),
    #                                                            batch_labels[task][0].to(torch.bool), colors="blue", alpha=0.5)
    #         Image.fromarray(semseg.numpy().transpose((1, 2, 0))
    #                         ).save(f'{path}/{id}_{task}_gt_masked.png')
    #
    #     elif task == "human_parts":
    #
    #         batch_predictions[task][0][batch_predictions[task][0] == 255] = 0
    #         vis_humans = vis_parts(batch_predictions[task][0].cpu())
    #         Image.fromarray(vis_humans).save(f'{path}/{id}_{task}_pred.png')
    #         parts_gt = batch_labels[task][:, 0, :, :]
    #         parts_gt[0][parts_gt[0] == 255] = 0
    #         vis_humans_gt = vis_parts(parts_gt[0].cpu().detach().to(torch.uint8))
    #         Image.fromarray(vis_humans_gt).save(f'{path}/{id}_{task}_gt.png')
    #
    #     elif task == "normals":
    #
    #         batch_label = batch_labels[task].cpu().detach().squeeze(axis=0).permute(1,2,0)
    #         normal_label = (F.normalize(batch_label, p=2, dim=2) + 1.0) * 255 / 2.0
    #         normal_label = normal_label.numpy()
    #         predictions[task] = normalize(predictions[task], 0, 255)
    #         Image.fromarray(normal_label.astype(np.uint8)).save(
    #             f'{path}/{id}_{task}_gt.png')
    #         Image.fromarray(predictions[task].astype(np.uint8)).save(
    #             f'{path}/{id}_{task}_pred.png')
    #
    #
    #     else:
    #         labels[task] = normalize(labels[task], 0, 255)
    #         predictions[task] = normalize(predictions[task], 0, 255)
    #         Image.fromarray(labels[task].astype(np.uint8)).save(
    #             f'{path}/{id}_{task}_gt.png')
    #         Image.fromarray(predictions[task].astype(np.uint8)).save(
    #             f'{path}/{id}_{task}_pred.png')


# def save_imgs_mtl(batch_imgs, batch_labels, batch_predictions, path, id):
#
#     # 수정할 부분 : 1) normalized gt, sal output,
#     import torchvision
#
#     imgs = tens2image(batch_imgs, transpose=True)
#     labels = {task: tens2image(label, transpose=True)
#               for task, label in batch_labels.items()}
#     predictions = {task: tens2image(prediction)
#                    for task, prediction in batch_predictions.items()}
#
#     Image.fromarray(normalize(imgs, 0, 255).astype(
#         np.uint8)).save(f'{path}/{id}_img.png')
#
#     for task in labels.keys():
#         if task == "semseg":
#             print(np.sum(labels[task] != 255))
#             labels[task] = labels[task] != 255
#             predictions[task] = predictions[task] != 225
#             batch_imgs = 255*(batch_imgs-torch.min(batch_imgs)) / \
#                 (torch.max(batch_imgs)-torch.min(batch_imgs))
#
#             batch_predictions[task][0][batch_predictions[task][0] == 255] = 0
#             vis_seg = vis_semseg(batch_predictions[task][0].cpu())
#             Image.fromarray(vis_seg).save(f'{path}/{id}_{task}_pred.png')
#             seg_gt = batch_labels[task][:, 0, :, :]
#             seg_gt[0][seg_gt[0] == 255] = 0
#             vis_seg_gt = vis_semseg(seg_gt[0].cpu().detach().to(torch.uint8))
#             Image.fromarray(vis_seg_gt).save(f'{path}/{id}_{task}_gt.png')
#
#             semseg = torchvision.utils.draw_segmentation_masks(batch_imgs[0].cpu().detach().to(torch.uint8),
#                                                                batch_predictions[task][0].to(torch.bool), colors="blue", alpha=0.5)
#             Image.fromarray(semseg.numpy().transpose((1, 2, 0))
#                             ).save(f'{path}/{id}_{task}_pred_masked.png')
#             semseg = torchvision.utils.draw_segmentation_masks(batch_imgs[0].cpu().detach().to(torch.uint8),
#                                                                batch_labels[task][0].to(torch.bool), colors="blue", alpha=0.5)
#             Image.fromarray(semseg.numpy().transpose((1, 2, 0))
#                             ).save(f'{path}/{id}_{task}_gt_masked.png')
#
#         elif task == "human_parts":
#
#             batch_predictions[task][0][batch_predictions[task][0] == 255] = 0
#             vis_humans = vis_parts(batch_predictions[task][0].cpu())
#             Image.fromarray(vis_humans).save(f'{path}/{id}_{task}_pred.png')
#             parts_gt = batch_labels[task][:, 0, :, :]
#             parts_gt[0][parts_gt[0] == 255] = 0
#             vis_humans_gt = vis_parts(parts_gt[0].cpu().detach().to(torch.uint8))
#             Image.fromarray(vis_humans_gt).save(f'{path}/{id}_{task}_gt.png')
#
#         elif task == "normals":
#
#             batch_label = batch_labels[task].cpu().detach().squeeze(axis=0).permute(1,2,0)
#             normal_label = (F.normalize(batch_label, p=2, dim=2) + 1.0) * 255 / 2.0
#             normal_label = normal_label.numpy()
#             predictions[task] = normalize(predictions[task], 0, 255)
#             Image.fromarray(normal_label.astype(np.uint8)).save(
#                 f'{path}/{id}_{task}_gt.png')
#             Image.fromarray(predictions[task].astype(np.uint8)).save(
#                 f'{path}/{id}_{task}_pred.png')
#
#
#         else:
#             labels[task] = normalize(labels[task], 0, 255)
#             predictions[task] = normalize(predictions[task], 0, 255)
#             Image.fromarray(labels[task].astype(np.uint8)).save(
#                 f'{path}/{id}_{task}_gt.png')
#             Image.fromarray(predictions[task].astype(np.uint8)).save(
#                 f'{path}/{id}_{task}_pred.png')




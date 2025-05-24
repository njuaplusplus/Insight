import argparse
import os
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from einops import rearrange
import ssl
from tqdm import tqdm
import lpips
import wandb

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config

from mist_utils import load_image_from_path


ssl._create_default_https_context = ssl._create_unverified_context
os.environ['TORCH_HOME'] = os.getcwd()
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hub/')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model_from_config(config, ckpt, verbose: bool = False):
    """
    Load model from the config and the ckpt path.
    :param config: Path of the config of the SDM model.
    :param ckpt: Path of the weight of the SDM model
    :param verbose: Whether to show the unused parameters weight.
    :returns: A SDM model.
    """
    print(f"Loading model from {ckpt}")

    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    # Support loading weight from NovelAI
    if "state_dict" in sd:
        import copy
        sd_copy = copy.deepcopy(sd)
        for key in sd.keys():
            if key.startswith('cond_stage_model.transformer') and not key.startswith('cond_stage_model.transformer.text_model'):
                newkey = key.replace('cond_stage_model.transformer', 'cond_stage_model.transformer.text_model', 1)
                sd_copy[newkey] = sd[key]
                del sd_copy[key]
        sd = sd_copy

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


class target_model(nn.Module):
    """
    A virtual model which computes the insight loss in forward function.
    """

    def __init__(self, model,
                 config,
                 condition: str
                 ):
        """
        :param model: A SDM model.
        :param config: The config of the model and the attack.
        :param condition: The condition for computing the semantic loss.
        """
        super().__init__()
        self.model = model
        self.condition = condition
        self.lpips_rate = config['lpips_rate']
        self.lpips_budget = config['lpips_budget']
        self.lnorm_rate = config['lnorm_rate']
        self.loss_semantic_rate = config['loss_semantic_rate']
        self.loss_encoder_rate = config['loss_encoder_rate']
        self.loss_watermark_rate = config['loss_watermark_rate']
        self.loss_watermark_semantic_rate = config['loss_watermark_semantic_rate']

        self.fn = nn.MSELoss(reduction="mean")
        self.loss_fn_alex = lpips.LPIPS(net='vgg').cuda()

        self.target_img = None  # the target image, e.g., the photo
        self.watermark_img = None  # the watermark image used as the negative sample

    def get_latent(self, x):
        """
        Compute the semantic loss and the encoded information of the input.
        :return: encoded info of x, semantic loss
        """
        z = self.model.get_first_stage_encoding(self.model.encode_first_stage(x)).to(device)
        return z

    def loss_semantic_fn(self, z, my_photo):
        """
        Compute the semantic loss between the input and the target.
        :param z: The encoded information of the input.
        :param my_photo: The photo image.
        :return: The semantic loss.
        """
        assert my_photo is not None
        c = self.model.get_learned_conditioning(self.condition)
        my_photo = self.model.get_first_stage_encoding(self.model.encode_first_stage(my_photo)).to(device)
        loss = self.model(z, c, my_photo=my_photo)[0]
        return loss

    def forward(self, x, step_i, img_id, org_x=None):
        """
        Compute the insight loss.
        """
        zx = self.get_latent(x)
        zy = self.get_latent(self.target_img)

        # compute semantic loss, ie. the UNet loss
        loss_semantic = self.loss_semantic_rate and self.loss_semantic_fn(zx, my_photo=self.target_img)

        # compute impress loss, ie. the lpips bound and the reconstruction loss
        _X_p = self.lnorm_rate and self.model.decode_first_stage(zx)
        impress_lnorm_loss = self.lnorm_rate and nn.functional.mse_loss(_X_p, x)
        impress_lpips_loss = self.lpips_rate and self.loss_fn_alex(x, org_x)

        # compute the encoder loss, ie. the VAE loss
        loss_encoder = self.loss_encoder_rate and self.fn(zx, zy)

        if self.loss_watermark_rate > 0. and self.watermark_img is not None:
            # compute the negative loss in the encoder
            watermark_zx = self.get_latent(self.watermark_img)
            # we want to maximize it
            loss_watermark = self.loss_encoder_rate and -self.fn(zx, watermark_zx)
        else:
            self.loss_watermark_rate = 0.
            loss_watermark = 0.
        
        if self.loss_watermark_semantic_rate > 0. and self.watermark_img is not None:
            # compute the negative loss in the UNet
            # we want to maximize the similarity between zx and watermark_zx
            loss_watermark_semantic = self.loss_semantic_rate and -self.loss_semantic_fn(zx, my_photo=self.watermark_img)
        else:
            self.loss_watermark_semantic_rate = 0.
            loss_watermark_semantic = 0.

        wandb.log(
            {
                f'encoder loss {img_id}': self.loss_encoder_rate and loss_encoder.item(),
                f'semantic loss {img_id}': self.loss_semantic_rate and loss_semantic.item(),
                f'lnorm loss {img_id}': self.lnorm_rate and impress_lnorm_loss.item(),
                f'lpips loss {img_id}': self.lpips_rate and impress_lpips_loss.item(),
                f'watermark loss {img_id}': self.loss_watermark_rate and loss_watermark.item(),
                f'watermark semantic loss {img_id}': self.loss_watermark_semantic_rate and loss_watermark_semantic.item(),
                'step_i': step_i,
            }
        )

        loss = ( self.loss_encoder_rate * loss_encoder
                + self.loss_semantic_rate * loss_semantic
                + self.lnorm_rate * impress_lnorm_loss
                + self.lpips_rate * max(impress_lpips_loss-self.lpips_budget, 0)
                + self.loss_watermark_rate * self.loss_encoder_rate * loss_watermark
                + self.loss_watermark_semantic_rate * self.loss_semantic_rate * loss_watermark_semantic
                )
        wandb.log(
            {
                f'loss {img_id}': loss.item(),
                'step_i': step_i,
            }
        )
        return loss


def init(my_config, object: bool = False, seed: int = 23, ckpt: str = None, base: str = None):
    """
    Prepare the config and the model used for purifying the adversarial noises.
    :param object: Set True if the targeted images describes a specifc object instead of a style.
    :returns: the wrapped model.
    """

    if ckpt is None:
        ckpt = 'models/ldm/stable-diffusion-v1/model.ckpt'

    if base is None:
        base = 'configs/stable-diffusion/v1-inference-attack.yaml'

    seed_everything(seed)
    imagenet_templates_small_style = ['a painting']
    imagenet_templates_small_object = ['a photo']

    config_path = os.path.join(os.getcwd(), base)
    config = OmegaConf.load(config_path)

    ckpt_path = os.path.join(os.getcwd(), ckpt)
    model = load_model_from_config(config, ckpt_path).to(device)

    if object:
        imagenet_templates_small = imagenet_templates_small_object
    else:
        imagenet_templates_small = imagenet_templates_small_style

    input_prompt = [imagenet_templates_small[0] for i in range(1)]
    net = target_model(model, my_config, input_prompt)
    net.eval()

    return net


def infer(net, config, img: PIL.Image.Image, tar_img: PIL.Image.Image, img_id, watermark_img=None) -> np.ndarray:
    """
    Process the protected image and generate the purified image.
    :param net: The model used for purifying the adversarial noises.
    :param config: config for the attack.
    :param img: The protected image.
    :param tar_img: The target image, e.g., the photo.
    :param img_id: The id of the image.
    :param watermark_img: The watermark image used as the negative sample.
    :returns: A purified image.
    """

    steps = config["steps"]
    opt_type = config['opt_type']
    trans = transforms.Compose([transforms.ToTensor()])

    assert img.size == tar_img.size == (512, 512), 'img and tar_img should have the same size'

    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    img = img[:, :, :3]
    tar_img = np.array(tar_img).astype(np.float32) / 127.5 - 1.0
    tar_img = tar_img[:, :, :3]

    if watermark_img is not None:
        watermark_img = np.array(watermark_img).astype(np.float32) / 127.5 - 1.0
        watermark_img = watermark_img[:, :, :3]
        watermark_img = trans(watermark_img).unsqueeze(0).to(device)
        net.watermark_img = watermark_img

    data_source = trans(img).unsqueeze(0).to(device)

    target_img = trans(tar_img).unsqueeze(0).to(device)
    net.target_img = target_img

    if opt_type == 'opt':
        # use optimizer
        attack_output = my_perturb_iterative(data_source, net, steps, img_id, clip_min=-1.0, clip_max=1.0, lr=config['opt_lr'], noise=0.1)
    else:
        # use pgd
        attack_output = pgd_perturb_iterative(data_source, net, steps, img_id, clip_min=-1.0, clip_max=1.0, epsilon=config["epsilon"], pgd_stepsize=config["pgd_stepsize"])

    output = attack_output[0]
    save_adv = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0).detach()
    grid_adv = 255. * rearrange(save_adv, 'c h w -> h w c').cpu().numpy()
    grid_adv = grid_adv
    return grid_adv


def my_perturb_iterative(xvar, net, steps, img_id, clip_min=0.0, clip_max=1.0, lr=0.01, noise=0.1):
    # this method uses optimizer instead of pgd fixed step.

    X_adv = xvar
    X_p = X_adv.detach().clone()  + (torch.randn_like(X_adv) * noise)
    X_p.requires_grad_(True)
    optimizer = torch.optim.Adam([X_p], lr=lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=1e-5)

    pbar = tqdm(range(steps))

    for step_i in pbar:
        optimizer.zero_grad()
        outputs = net(X_p, step_i, img_id, org_x=X_adv)
        loss = outputs
        loss.backward()

        optimizer.step()
        scheduler.step()

        X_p.data = torch.clamp(X_p, min=clip_min, max=clip_max)
        pbar.set_description(f"[Running purify]: Loss: {outputs.item():.5f}")

    return X_p.detach().clone()


def pgd_perturb_iterative(xvar, net, steps, img_id,
                          clip_min=0.0, clip_max=1.0, epsilon=16, pgd_stepsize=1):
    ord = np.inf

    epsilon = epsilon/255.0 * (clip_max-clip_min)
    pgd_stepsize = pgd_stepsize/255.0 * (clip_max-clip_min)

    pbar = tqdm(range(steps))

    delta = torch.zeros_like(xvar)
    delta.data.uniform_(-1., 1.)
    delta.data = epsilon * delta.data
    delta.data = torch.clamp(xvar + delta.data, min=clip_min, max=clip_max) - xvar
    delta.requires_grad_()

    for step_i in pbar:
        outputs = net(xvar + delta, step_i, img_id, org_x=xvar)

        loss = outputs
        loss.backward()

        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data - pgd_stepsize * grad_sign
            delta.data = torch.clamp(delta.data, min=-epsilon, max=epsilon)
            delta.data = torch.clamp(xvar.data + delta.data, min=clip_min, max=clip_max) - xvar.data
        else:
            error = "Only ord = inf have been implemented"
            raise NotImplementedError(error)
        delta.grad.data.zero_()

        pbar.set_description(f"[Running purify]: Loss: {outputs.item():.5f}")

    x_adv = torch.clamp(xvar + delta, min=clip_min, max=clip_max)

    return x_adv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vangogh",
        help="path of output dir",
    )
    parser.add_argument(
        "--photo_dir",
        type=str,
        required=True,
        help="path of photo dir",
    )
    parser.add_argument(
        "-inp",
        "--input_dir_path",
        type=str,
        default=None,
        help="Path of the dir of images to be processed.",
    )
    parser.add_argument(
        "--opt_type",
        type=str,
        default='pgd',
        choices=['pgd', 'opt'],
        help="Optimization type. Default is pgd."
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=int,
        default=16,
        help=(
            "The bound of pgd."
        ),
    )
    parser.add_argument(
        "--pgd_stepsize",
        type=int,
        default=1,
        help=(
            "The step size of pgd. Only used when opt_type == pgd."
        ),
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=100,
        help=(
            "The optimization steps."
        ),
    )
    parser.add_argument(
        "-in_size",
        "--input_size",
        type=int,
        default=512,
        help=(
            "The input_size of images"
        ),
    )
    parser.add_argument(
        "--opt_lr",
        type=float,
        default=0,
        help="Optimizer's learning rate. This is only used when opt_type == opt."
    )
    parser.add_argument(
        "--lpips_budget",
        type=float,
        default=0,
        help="LPIPS budget"
    )
    parser.add_argument(
        "--lpips_rate",
        type=float,
        default=0,
        help="LPIPS rate"
    )
    parser.add_argument(
        "--lnorm_rate",
        type=float,
        default=0,
        help="lnorm rate"
    )
    parser.add_argument(
        "--loss_encoder_rate",
        type=float,
        default=1.,
        help="loss encoder rate"
    )
    parser.add_argument(
        "--loss_semantic_rate",
        type=float,
        default=0.1,
        help="loss semantic rate"
    )
    parser.add_argument(
        "--loss_watermark_rate",
        type=float,
        default=0.,
        help="loss watermark rate"
    )
    parser.add_argument(
        "--loss_watermark_semantic_rate",
        type=float,
        default=0.,
        help="loss watermark semantic rate"
    )
    args = parser.parse_args()
    return args


def compose_dirname_suffix(config):
    return '_'.join([str(v) for v in config.values()])


if __name__ == "__main__":
    args = parse_args()

    config={
        'opt_type': args.opt_type,
        'epsilon': args.epsilon,
        'pgd_stepsize': args.pgd_stepsize,
        'steps': args.steps,
        'input_size': args.input_size,
        'opt_lr': args.opt_lr,
        'lpips_budget': args.lpips_budget,
        'lpips_rate': args.lpips_rate,
        'lnorm_rate': args.lnorm_rate,
        'loss_encoder_rate': args.loss_encoder_rate,
        'loss_semantic_rate': args.loss_semantic_rate,
        'loss_watermark_rate': args.loss_watermark_rate,
        'loss_watermark_semantic_rate': args.loss_watermark_semantic_rate,
    }

    wandb.init(
        entity="answ",
        project="attack_mist",
        name=f"{args.output_dir}_{compose_dirname_suffix(config)}",
        config=config
    )

    input_size = config['input_size']
    
    # The directory of the photos
    photo_dir = args.photo_dir

    # The directory of the protected images.
    image_dir_path = args.input_dir_path

    # Init the wrapped model.
    net = init(config)

    # The directory of the output images.
    output_dir = os.path.join('outputs/dirs', args.output_dir)
    output_path_dir = output_dir + '_' + compose_dirname_suffix(config)
    if not os.path.exists(output_path_dir):
        os.makedirs(output_path_dir)
    else:
        # raise AssertionError(f'output_path_dir alreay exists')
        print(f'output_path_dir alreay exists')

    for img_id in sorted(os.listdir(image_dir_path)):

        if os.path.splitext(img_id)[1].lower().strip('.') not in ['png', 'jpg', 'jpeg', ]:
            print(f'skip non-image file {img_id}')
            continue

        # Find the corresponding image in the photo directory
        target_image_path = os.path.join(photo_dir, img_id)
        if not os.path.isfile(target_image_path):
            target_image_path = os.path.join(photo_dir, os.path.splitext(img_id)[0]+'.jpg')

            if not os.path.isfile(target_image_path):
                target_image_path = os.path.join(photo_dir, os.path.splitext(img_id)[0]+'.jpeg')

                if not os.path.isfile(target_image_path):
                    print(f'{target_image_path} does not exist')
                    continue

        output_path = os.path.join(output_path_dir, img_id)
        if os.path.exists(output_path):
            print(f'{output_path} already exist, so skip')
            continue

        img_path = os.path.join(image_dir_path, img_id)

        print('img_path', img_path)
        print('target_img_path', target_image_path)

        image = load_image_from_path(img_path, input_size)
        target_img = load_image_from_path(target_image_path, input_size)

        watermark_image_path = img_path
        watermark_img = load_image_from_path(watermark_image_path, input_size)

        output_image = infer(net, config, image, target_img, img_id, 
                             watermark_img=watermark_img)

        output = Image.fromarray(output_image.astype(np.uint8))
        print("Output image saved in path {}".format(output_path))
        output.save(output_path)

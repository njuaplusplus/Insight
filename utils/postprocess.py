import os 
import PIL
from PIL import Image
import torchvision.transforms.functional as tv


def crop_resize_from_path(input_path, input_size, target_size):
    crop = (input_size - target_size)//2
    box = [crop, crop, 512 - crop, 512-crop]
    target_path = input_path + '_crop_resize'
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    for img_id in os.listdir(input_path):
        input_image_path = os.path.join(input_path, img_id)
        if os.path.isdir(input_image_path):
            continue
        img = Image.open(input_image_path).resize((input_size, input_size),
                                                  resample=PIL.Image.BICUBIC)
        img = img.crop(box).resize((input_size, input_size),
                                   resample=PIL.Image.BICUBIC)
        target_image_path = os.path.join(target_path, img_id)
        img.save(target_image_path)


def my_crop_resize(input_path):
    """
    input_path is a folder, and the saved folder will be input_path+'_crop_resize'
    """
    crop_resize_from_path(input_path, 512, 384)


def my_jpeg_compression(input_path, quality=15):
    input_size = 512
    target_path = input_path + f'_jpeg_compress_{quality}'
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    for img_id in os.listdir(input_path):
        input_image_path = os.path.join(input_path, img_id)
        if os.path.isdir(input_image_path):
            continue
        img = Image.open(input_image_path).resize((input_size, input_size),
                                                  resample=PIL.Image.BICUBIC)
        target_image_path = os.path.join(target_path, f'{img_id}.jpg')
        img.save(target_image_path, quality=quality)


def add_gaussian(input_path, kernel_size=8):
    input_size = 512
    target_path = input_path + f'_gaussian_{kernel_size}'
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    for img_id in os.listdir(input_path):
        if 'jsonl' in img_id:
            continue
        input_image_path = os.path.join(input_path, img_id)
        if os.path.isdir(input_image_path):
            continue
        img = Image.open(input_image_path).resize((input_size, input_size),
                                                  resample=PIL.Image.BICUBIC)
        img = tv.gaussian_blur(img,[1,1], [kernel_size, kernel_size]) 
        target_image_path = os.path.join(target_path, img_id)
        img.save(target_image_path)


if __name__ == "__main__":
    # input_path = 'test/vangogh_16_100_512_1_2_1'
    # input_path = '/data/siyuan_cheng/an/diffusers/examples/dreambooth/mist/mist'
    # input_path = '/data/siyuan_cheng/an/photoguard/notebooks/source'
    # input_path = '/data/siyuan_cheng/an/photoguard/notebooks/simple_pgd'
    # input_path = '/data/siyuan_cheng/an/diffusers/examples/dreambooth/glaze/glaze'
    # crop_resize_from_path(input_path, 512, 384)
    input_paths = [
                    # '/data4/user/an93/Diff-Protect/out/sds_eps16_steps100_gmode+/source/attacked',
                    # '/data4/user/an93/Diff-Protect/out/sds_eps16_steps100_gmode-/source/attacked',
                    # '/data4/user/an93/Diff-Protect/out/sdsT5_eps16_steps100_gmode-/source/attacked',
                    # '/data4/user/an93/Diff-Protect/out/advdm_eps16_steps100_gmode+/source/attacked',
                    # '/data4/user/an93/Diff-Protect/out/advdm_eps16_steps100_gmode-/source/attacked',
                    # '/data4/user/an93/helen_dreambooth/mist/raphael_16_100_512_1_2_1_0_0',
                    # '/data4/user/an93/original_mist/outputs/dirs/albrecht_16_100_512_1_2_1_0_0',
                    # '/data4/user/an93/original_mist/outputs/dirs/camille_16_100_512_1_2_1_0_0',
                    # '/data4/user/an93/original_mist/outputs/dirs/childe_16_100_512_1_2_1_0_0',
                    # '/data4/user/an93/original_mist/outputs/dirs/paul_16_100_512_1_2_1_0_0',
                    # '/data4/user/an93/original_mist/outputs/dirs/pyotr_16_100_512_1_2_1_0_0',
                    # '/data4/user/an93/helen_face/selected/adv_l2_eps16_step1_iter200grad_reps10_eta1_diff_steps4_guidance7.5_seed0',
                    # '/data4/user/an93/helen_face/mask',
                    '/data4/user/an93/our_diff_attack_data/mist/outputs/dirs/mist_google_drive',
                ]
    # for artist in [
    #               "raphael_kirchner",
    #               "camille_pissarro",
    #               "pyotr_konchalovsky",
    #               "childe_hassam",
    #               "paul_cezanne",
    #               "albrecht_durer",
    #              ]:
    #     for attack in [
    #            "advdm_eps16_steps100_gmode+",
    #            "advdm_eps16_steps100_gmode-",
    #            "sds_eps16_steps100_gmode+",
    #            "sds_eps16_steps100_gmode-",
    #            "sdsT5_eps16_steps100_gmode-",
    #        ]:
    #         input_paths.append(f'/data4/user/an93/Diff-Protect/protected_imgs/{artist}/{attack}/attacked')

    # for artist in [
    #         'albrecht-durer',
    #         'camille-pissarro',
    #         'childe-hassam',
    #         'paul-cezanne',
    #         'pyotr-konchalovsky',
    #         'raphael-kirchner',
    #         'vincent-van-gogh',
    #         ]:
    #     input_paths.append(f'/data4/user/an93/yanlu/processed_wikiart/{artist}/adv_p0.05_alpha30_iter500_lr0.01/train/trans_Cubism_by_Picasso_seed0')

    print('\n'.join(input_paths))
    for input_path in input_paths:
        my_jpeg_compression(input_path)
        # my_crop_resize(input_path)
        add_gaussian(input_path)

# Official Code for Insight (USENIX Security 2024)

This is the PyTorch implementation for the USENIX Security 2024 paper ["Rethinking the Invisible Protection against Unauthorized Image Usage in Stable Diffusion"](https://www.usenix.org/conference/usenixsecurity24/presentation/an).


## Environment

We build our code based on [Mist](https://github.com/psyker-team/mist), so we use a similar conda environment. To install the environment:

```
conda env create -f env_insight.yml
conda activate insight
```

We use the Stable-diffusion-model v1.4 checkpoint by default, which can be downloaded as:

```
wget -c https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
mkdir -p  models/ldm/stable-diffusion-v1
mv sd-v1-4.ckpt models/ldm/stable-diffusion-v1/model.ckpt
```

## Running examples

### 1. Prepare the protected images

We may put the protected images into any directories. For example, we downloaded the samples provided by the Mist repo into [`data_samples/mist/mist_google_drive/`](data_samples/mist/mist_google_drive/).

### 2. Prepare the photos

We took photos of the protected images using an iPhone. Then, we used the Photos app's built-in image editing functions to rotate and crop the photo and resolve the distortion. 
You may also try different apps, such as Scanner Pro or CamScanner. A high-PPI monitor (e.g., 27/24 4k) is recommended. Otherwise, the photo may contain noisy strips or dots of the monitor's LED array.
We renamed the photos accordingly and put them into [`data_samples/photos/vangogh/`](data_samples/photos/vangogh/).

### 3. Run Insight to align the protected images with the photos

```
zsh run_insight_mist.zsh
```

### 4. Tips on tuning the parameters

Take style mimicry as an example. The attackers can first use the (protected) images to train a model to generate some images. 
If they find the generated images have a completely different style or an obvious watermark pattern, they can infer that the protection may constrain VAE and thus use larger weights for the VAE-related loss. 
If they find the generated images have chaotic contents or color schemes, the protection may constrain the UNet, and they can use larger weights for the UNet-related loss.


## BibTex

Please cite our work as follows for any purpose of usage.

```
@inproceedings {An.Insight.Security.2024,
    author = {Shengwei An and Lu Yan and Siyuan Cheng and Guangyu Shen and Kaiyuan Zhang and Qiuling Xu and Guanhong Tao and Xiangyu Zhang},
    title = {Rethinking the Invisible Protection against Unauthorized Image Usage in Stable Diffusion},
    booktitle = {33nd USENIX Security Symposium (USENIX Security 24)},
    year = {2024},
    publisher = {USENIX Association},
}
```
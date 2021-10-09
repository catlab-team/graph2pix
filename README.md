# Graph2Pix: A Graph-Based Image to Image Translation Framework

## Installation

Install the dependencies in ``env.yml``
``` bash
$ conda env create -f env.yml
$ conda activate pix-env
```

## Examples
Train ArtBreeder dataset with Smart Disc Feature enabled and 7 disc images.
``` bash
$ python train.py --dataroot ./datasets/newbreeder/ --name art_newbreeder_allparents_bs8_disc --no_instance --fineSize 256 --loadSize 256 --label_nc 0 --resize_or_crop resize_and_crop --output_nc 3 --batchSize 8 --mv --smart_disc --num_disc_images 7
```

Generate images from trained model.
``` bash
$ python gen_imgs.py --dataroot ./datasets/newbreeder/ --name art_newbreeder_allparents_bs8_disc --no_instance --fineSize 256 --loadSize 256 --label_nc 0 --resize_or_crop resize_and_crop --output_nc 3 --batchSize 8 --mv --smart_disc --num_disc_images 7 --which_epoch 50
```

FID Calculation
``` bash
$ python -m pytorch_fid gen_images/art_newbreeder_allparents_bs8_disc/gts gen_images/art_newbreeder_allparents_bs8_disc/generated/
```

## Acknowledgments
The basis of this code is [pix2pixHD](https://github.com/NVIDIA/pix2pixHD).

## Citation

If you use this code for your research, please cite our paper:
```
@article{gokay2021graph2pix,
  title={Graph2Pix: A Graph-Based Image to Image Translation Framework},
  author={Gokay, Dilara and Simsar, Enis and Atici, Efehan and Ahmetoglu, Alper and Yuksel, Atif Emre and Yanardag, Pinar},
  journal={arXiv preprint arXiv:2108.09752},
  year={2021}
}
```

# Face Attribute Manipulation

Using the AttGAN implementation of [elvisyjlin/AttGAN-PyTorch/](https://github.com/elvisyjlin/AttGAN-PyTorch/)


## Data
* Dataset
  * [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset
    * [Images](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADSNUu0bseoCKuxuI5ZeTl1a/Img?dl=0&preview=img_align_celeba.zip) should be placed in `./data/img_align_celeba/*.jpg`
    * [Attribute labels](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAA8YmAHNNU6BEfWMPMfM6r9a/Anno?dl=0&preview=list_attr_celeba.txt) should be placed in `./data/list_attr_celeba.txt`
    
## Pretrained Model
Download the model '256_shortcut1_inject1_none' and put it under 'output/'
Link: [Pretrained Model](https://drive.google.com/drive/folders/1_E5YCb4XOTZpt6KBwBzSaJdofoqPViN8) 

## Usage
* Put the 'img_align_celeba' folder under '../dataset/img_align_celeba'
* Put the textfile 'list_attr_celeba.txt' under '../dataset/list_attr_celeba.txt'
* Run demo.sh

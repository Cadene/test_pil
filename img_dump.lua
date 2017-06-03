require 'image'
require 'torch'

--path = '/local/common-data/imagenet_2012/images/val/n04447861/ILSVRC2012_val_00028306.JPEG'
path = 'lena.png'
img = image.load(path,
		3, 'float')
torch.save('img.t7', img)

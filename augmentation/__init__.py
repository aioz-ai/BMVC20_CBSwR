from torchvision import transforms
from augmentation.auto_augment import AutoAugment

normalize = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda x: x * 255.0),
    transforms.Normalize(mean=[122.7717, 115.9465, 102.9801], std=[1, 1, 1]),
    transforms.Lambda(lambda x: x[[2, 1, 0], ...])
])


transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(size=224),
    transforms.RandomHorizontalFlip(p=0.75),
    normalize,
    ])


transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(224),
    normalize,
    ])

transform_aug = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomRotation((-10, 10)),
    transforms.CenterCrop(224)])
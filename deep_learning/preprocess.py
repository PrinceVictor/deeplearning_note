from torchvision import transforms

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

def data_argument(input_size, normalize):
    para_list = [
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-45, 45)),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ]
    return transforms.Compose(para_list)

def data_resize(input_size, normalize):
    para_list = [
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ]
    return transforms.Compose(para_list)

def get_transform(input_size = None, normalize = None, argument=False):
    normalize = __imagenet_stats

    if argument == True:
        return data_argument(input_size, normalize)
    else:
        return data_resize(input_size, normalize)
model = dict(
    arch='efficientnet_b0',
    pretrained=True,
    num_classes=206
)
device_ids = [0]
workers = 8
batch_size = 64
optimizer = dict(
    learning_rate=0.025,
    momentum=0.9,
    weight_decay=1e-4,
)
data='/data-input/mul_dimention/input' #change the path
annotations = dict(
    train=f'{data}/annotations/scene_train.txt', #change the path
    val=f'{data}/annotations/scene_train.txt', #change the path
    test=f'{data}/annotations/scene_val_without_label.txt' #change the path
)


lr_config = dict(
    warmup_epochs=5,
    warmup_ratio=0.001
)

log_config = dict(
    print_freq=10
)

# runtime settings 
total_epochs = 40
start_epoch = 0
resume = ''
evaluate = False
work_dir = f'./work_dirs/{model["arch"]}/'

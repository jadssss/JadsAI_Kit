DEFAULT_CONFIG = {
    'conv_layers': [{'filters': 32, 'kernel': 3, 'stride': 1, 'padding': 'same'}],
    'dense_layers': [128],
    'dropout': 0.2,
    'batch_norm': False,
    'activation': 'relu',
    'optimizer': 'adam',
    'lr': 0.001,
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy']
}

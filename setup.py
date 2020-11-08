from setuptools import setup, find_packages

setup(
  name = 'vit-pytorch-lightning',
  packages = find_packages(exclude=['examples']),
  version = '',
  license= = 'MIT',
  description = 'Vision Transformer(ViT) - PyTorch Lightning',
  author = 'Makoto Sofue',
  author_email = 'makoto2410.10@gmail.com',
  url = '',
  kerwords = [
    'artifical intelligence',
    'computer vision',
    'pytorch lightning'
  ],
  install_requires = [
    'torch>=1.6',
    'pytorch-lightning>=1.0.3'
  ],
  classifiers = [
    'Programmin Language :: Python :: 3.8'
  ]
)

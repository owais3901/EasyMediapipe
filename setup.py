from distutils.core import setup

setup(
    name='EasyMediapipe',
    version='0.0.1',
    license='MIT',
    description='',
    author='owais3901',
    author_email='siddiquiowais390@gmail.com',
    url='https://github.com/owais3901/EasyMediapipe.git',
    keywords=['Computer Vision','Mediapipe'],
    install_requires=[
        'mediapipe',
        'numpy',
        'opencv-python'
    ],
    python_requires='>=3.11',
)
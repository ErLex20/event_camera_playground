import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'dsec_publisher'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='neo',
    maintainer_email='alessandro.cretu2000@gmail.com',
    description='Replays DSEC event-camera HDF5 recordings as EventPacket messages for EVO.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'dsec_publisher = dsec_publisher.dsec_publisher_node:main',
        ],
    },
)

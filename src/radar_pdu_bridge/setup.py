from setuptools import setup

package_name = 'radar_pdu_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='1984027855@qq.com',
    description='Temporary ROS2 bridge for GNU Radio PDU broadcast',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'pdu_subscriber_node = radar_pdu_bridge.pdu_subscriber_node:main',
            'shm_pdu_broadcast_node = radar_pdu_bridge.shm_pdu_broadcast_node:main',
            'radar_decision_sync_shm_node = radar_pdu_bridge.radar_decision_sync_shm_node:main',
        ],
    },
)

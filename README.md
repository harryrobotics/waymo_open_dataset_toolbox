
You need to install:

numpy

waymo open dataset development kit (https://github.com/waymo-research/waymo-open-dataset

```
pip3 install waymo-open-dataset-tf-2-1-0==1.2.0 --user
```


You need to download the dataset from Waymo Open Dataset (Warning: about 1TB)  and the folder structure is like this:

```
waymo
├── raw_data
│   ├── segment-....
├── LICENSE
```

Example:

Lidar with ground truth bounding boxes (Need to add color for each class)

![](https://github.com/harryrobotics/waymo_open_dataset_toolbox/blob/master/media/lidar.png)

![](https://github.com/harryrobotics/waymo_open_dataset_toolbox/blob/master/media/lidar1.png)

Front camera:

![](https://github.com/harryrobotics/waymo_open_dataset_toolbox/blob/master/media/front_camera.png)

5 cameras:

![](https://github.com/harryrobotics/waymo_open_dataset_toolbox/blob/master/media/all_cameras.png)


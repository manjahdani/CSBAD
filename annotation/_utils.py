def df_to_txt(data, H, W, amalgamize_truck=True):  # class x_center y_center width height
    xmin = data['xmin']
    ymin = data['ymin']
    xmax = data['xmax']
    ymax = data['ymax']
    extract = data.loc[:, ['class', 'xmin', 'ymin', 'xmax', 'ymax']]
    if amalgamize_truck:
        extract.loc[extract['class'] == 7, 'class'] = 2
    extract['xmin'] = ((xmax + xmin) / 2) / W  # x_center
    extract['ymin'] = ((ymax + ymin) / 2) / H  # y_center
    extract['xmax'] = (xmax - xmin) / W  # width
    extract['ymax'] = (ymax - ymin) / H  # width
    return extract


class VideoLengthException(Exception):
    pass

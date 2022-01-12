import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

def main():
    # getting points from manual labeling (instance segmentation) and storing in a dataframe
    manual_instance_segmentation_annotation_df = pd.DataFrame()

    json_files = []

    for edit_path in ['traffic_signs_wrong_transform', 'traffic_signs_todo']:
        for group in ['1', '2', '3']:
            path_to_json = f'../../../data/shared/mapillary_vistas/training/hand_annotated_signs/{edit_path}/{group}/'
            curr_json_files = [path_to_json + pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]  
            json_files.extend(curr_json_files)

    for edit_path in ['traffic_signs_final_check']:
        for group in ['1']:
            path_to_json = f'../../../data/shared/mapillary_vistas/training/hand_annotated_signs/{edit_path}/{group}/'
            curr_json_files = [path_to_json + pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]  
            json_files.extend(curr_json_files)

    df_filenames = []
    df_points = []
    for json_path in json_files:
        filename = json_path.split('/')[-1].split('.json')[0] + '.png'
        df_filenames.append(filename)
        # try:
        with open(json_path) as f:
            json_data = json.load(f)
        # except:
        #     print(json_path)
        #     print(edit_path)
        #     qqq
        
        if len(json_data['shapes']) != 1:
            print(json_path)
        assert len(json_data['shapes']) == 1
        for annotation in json_data['shapes']:
            df_points.append(annotation['points'])

    manual_instance_segmentation_annotation_df['filename'] = df_filenames
    manual_instance_segmentation_annotation_df['points'] = df_points

    # loading df with manual class annotations
    manual_annotated_df = pd.read_csv('../../../data/shared/mtsd_v2_fully_annotated/traffic_sign_annotation_train.csv')

    # merging with instance labeling df
    manual_annotated_df = manual_annotated_df.merge(manual_instance_segmentation_annotation_df, left_on='filename', right_on='filename', how='left')

    CLASS_LIST = [
        'circle-750.0',
        'triangle-900.0',
        'triangle_inverted-1220.0',
        'diamond-600.0',
        'diamond-915.0',
        'square-600.0',
        'rect-458.0-610.0',
        'rect-762.0-915.0',
        'rect-915.0-1220.0',
        'pentagon-915.0',
        'octagon-915.0',
        'other-0.0-0.0',
    ]

    # relabeling shapes in df
    final_shapes = []
    for index, row in manual_annotated_df.iterrows():
        # if reannotated then choose our annotation
        if not np.isnan(row['new_class']):
            final_shapes.append(CLASS_LIST[int(row['new_class'])])
        # if group 1 then choose agreed shape
        elif row['group'] == 1:
            final_shapes.append(row['predicted_class'])
        elif row['group'] == 2:
            # resnet mostly correct on group 2
            final_shapes.append(row['predicted_class'])
        elif row['group'] == 3:
            # use annotation: if no annotation, we do not know the traffic sign and it is actually other
            final_shapes.append('other-0.0-0.0')

        # if shape is rect and use_rect flag is 1 then we use use regular src, tgt
        # if shape is NOT rect and use_rect flag is 1 then we

    manual_annotated_df['final_shape'] = final_shapes

    # read df with tgt, alpha, beta and merging
    df = pd.read_csv('mapillaryvistas_data.csv')

    manual_annotated_df['filename_x'] = manual_annotated_df['filename'].apply(lambda x: '_'.join(x.split('.png')[0].split('_')[:-1]) + '.jpg')
    manual_annotated_df = manual_annotated_df.merge(df, on=['filename_x', 'object_id'], how='left')

    final_df = manual_annotated_df[['filename', 'object_id', 'shape_x', 'predicted_shape_x', 'predicted_class_x', 'group_x', 'batch_number_x',
                        'row_x', 'column_x', 'new_class', 'todo', 'use_rect_for_contour', 'wrong_transform',
                        'use_polygon', 'occlusion', 'final_shape', 'points', 'tgt', 'alpha', 'beta', 'filename_png',
                        'xmin', 'ymin', 'xmin_ratio', 'ymin_ratio']]

    # final_df['tgt_final'] = final_df.apply(lambda x: x['points'] if (x['points'].any() or not np.isnan(x['points'])) else x['tgt'], axis=1)
    final_df['tgt_final'] = final_df.apply(lambda x: x['points'] if not isinstance(x['points'], float) else x['tgt'], axis=1)


    from ast import literal_eval
    # final_df["tgt_final"] = final_df["tgt_final"].apply(literal_eval)

    shape_df = pd.read_csv('shape_df.csv')

    # manual_annotated_df = manual_annotated_df.merge(shape_df, left_on=['filename_png', 'object_id'], right_on=['filename_png', 'object_id'], how='left')
    final_df = final_df.merge(shape_df, on=['filename_png', 'object_id'], how='left')

    final_df = final_df[final_df['occlusion'].isna()]

    tgt_final_values = []

    # TODO: remove. only used for debugging
    errors = []

    indices = []

    for index, row in tqdm(final_df.iterrows()):
        shape = row['final_shape'].split('-')[0]
        # curr_tgt = row['tgt_final'].apply(literal_eval)
        try:
            curr_tgt = literal_eval(row['tgt_final'])
            curr_tgt = np.array(curr_tgt)
        except:
            # print(row['tgt_final'])
            # print(type(row['tgt_final']))
            # qqq
            pass
        
        if not isinstance(row['points'], float):
            # tgt_final_values.append(np.array(curr_tgt))
            
            tgt_final_values.append(row['points'])

            # tgt_final_values.append(curr_tgt)
            # print(type(row['points']))
            # print(type(row['points']))
            # qqq
            # tgt_final_values.append(str(curr_tgt))
            # print(curr_tgt)
            # qqq
            # tgt_final_values.append(curr_tgt)
            continue
        
        offset_x_ratio = row['xmin_ratio']
        offset_y_ratio = row['ymin_ratio']
            
        h0, w0, h_ratio, w_ratio, w_pad, h_pad = row['h0'], row['w0'], row['h_ratio'], row['w_ratio'], row['w_pad'], row['h_pad']

        # Have to correct for the padding when df is saved (TODO: this should be simplified)
        pad_size = int(max(h0, w0) * 0.25)
        x_min = offset_x_ratio * (w0 + pad_size * 2) - pad_size
        y_min = offset_y_ratio * (h0 + pad_size * 2) - pad_size

        # Order of coordinate in tgt is inverted, i.e., (x, y) instead of (y, x)
        curr_tgt[:, 1] = (curr_tgt[:, 1] + y_min) * h_ratio + h_pad
        curr_tgt[:, 0] = (curr_tgt[:, 0] + x_min) * w_ratio + w_pad

        # if row['filename_x'] == '0KohgmStOYkLZM6v-Frfew_43.png':
            # print(curr_tgt)
            # print(row['tgt'])
            # print()
            # print(row)
            # qqq

        tgt_final_values.append(curr_tgt.tolist())

        if shape in ['triangle', 'triangle_inverted']:
            # assert len(curr_tgt) == 3
            if len(curr_tgt) != 3:
                # print(row)
                errors.append(row)
                indices.append(index)
                # qqq
        elif shape in ['square', 'diamond', 'octagon', 'circle', 'pentagon', 'rect']:
            # assert len(curr_tgt) == 4
            if len(curr_tgt) != 4:
                # print(row)
                errors.append(row)
                indices.append(index)
                # print('here', row.index)
                # qqq
                # qqq

        # tgt_final_values.append(curr_tgt)
        # print(type(curr_tgt))
    print('num errors', len(errors))

    error_df = final_df.loc[indices]
    print(error_df.shape)

    error_df['final_check'] = 1
    error_df['group'] = 1
    error_df['filename'] = error_df['filename_x']
    error_df.to_csv('error_df_2.csv', index=False)
    

    # qqq

    final_df['tgt_final'] = tgt_final_values


    final_df =final_df.rename(columns={
                                'shape_x': 'shape', 'predicted_shape_x': 'predicted_shape',
                                'predicted_class_x': 'predicted_class', 'batch_number_x': 'batch_number',
                                'row_x': 'row', 'column_x': 'column'
                            })

    final_df.to_csv('mapillary_vistas_final_merged.csv', index=False)

if __name__  == '__main__':
    main()


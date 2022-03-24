import pandas as pd
import os
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Save images for manual annotation', add_help=False)
    parser.add_argument('--column', default='todo', type=str)
    parser.add_argument('--group', default=1, type=int)
    return parser

def main(args, corrections_df):
    # df = pd.read_csv('../../../../data/shared/mtsd_v2_fully_annotated/traffic_sign_annotation_train.csv')
    # df = pd.read_csv('../../../../data/shared/mtsd_v2_fully_annotated/traffic_sign_annotation_validation.csv')
    df = pd.read_csv('error_df_validation.csv')
    # column = args.column
    column = 'final_check'
    df = df[df['occlusion'].isna()]
    # df = df[df[f'{column}'] == 1]
    # df = df[df['group'] == args.group]

    corrections_df = pd.concat([corrections_df, df], axis=0)
    
    print('number of images to label', len(df), '\n')

    # data_dir = '/data/shared/mapillary_vistas/training/'
    data_dir = '/data/shared/mapillary_vistas/validation/'

    img_path_cropped = os.path.join(data_dir, 'traffic_signs')
    img_path = os.path.join(data_dir, 'images')

    for filename in df['filename'].values:
        new_filename = '_'.join(filename.split('_')[:-1]) + '.jpg'

        img_file_cropped = os.path.join(img_path_cropped, filename)
        img_file = os.path.join(img_path, new_filename)

        # img_file_destination = f'/data/shared/mapillary_vistas/training/traffic_signs_{column}/{args.group}/'
        # img_file_destination_cropped = f'/data/shared/mapillary_vistas/training/traffic_signs_{column}/{args.group}_cropped/'
        img_file_destination = f'/data/shared/mapillary_vistas/validation/traffic_signs_{column}/{args.group}/'
        img_file_destination_cropped = f'/data/shared/mapillary_vistas/validation/traffic_signs_{column}/{args.group}_cropped/'

        if not os.path.exists(img_file_destination):
            os.makedirs(img_file_destination)

        if not os.path.exists(img_file_destination_cropped):
            os.makedirs(img_file_destination_cropped)

        if not os.path.isfile(img_file):
            print(filename)
            print(new_filename)
            print(img_file)
            raise Exception()

        img_file_destination += filename
        img_file_destination_cropped += filename
        os.system(f'cp {img_file} {img_file_destination}') 
        os.system(f'cp {img_file_cropped} {img_file_destination_cropped}') 

    # corrections_df.to_csv("mapillary_vistas_corrections.csv", index=False)
    corrections_df.to_csv("mapillary_vistas_corrections_validation.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Save images for manual annotation', parents=[get_args_parser()])
    args = parser.parse_args()

    try:
        # corrections_df = pd.read_csv("mapillary_vistas_corrections.csv")
        corrections_df = pd.read_csv("mapillary_vistas_corrections_validation.csv")
    except:
        corrections_df = pd.DataFrame()

    main(args, corrections_df)
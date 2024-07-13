import os
from skimage.io import imread

from .phenomics import PlateNormalize
from .verbosity import Verbosity
from joblib import Parallel, delayed


import logging
from datetime import datetime

logging.basicConfig(filename=f'Phenomics/logging/{datetime.today().strftime("%Y-%m-%d")}.log'
                    ,format='[%(asctime)s | %(levelname)s ] %(message)s', level=logging.INFO)

def analysis_pipeline(fpath_img_dataset, fpath_output, n_jobs=4 ):
    '''
    fpath_img_dataset: "/user/path/to/dataset/" Must include separator at the end
    '''
    fpath_imgs = os.listdir(fpath_img_dataset)
    for path in fpath_imgs:
        path = fpath_img_dataset + path
    bad_analysis_log = open(f"{fpath_output}invalid_imgs.log", mode='w')
    def analyze_phenotype(fpath_dataset, fpath_img):
        fpath_curr_img = fpath_dataset + fpath_img
        try:
            sample_name = fpath_curr_img.split('.')[0]
            logging.info(f"Starting Sample: {fpath_curr_img}")
            verb = Verbosity(True)
            verb.start(f"analysis of sample: {sample_name}")
            img = imread(fpath_curr_img)
            data = PlateNormalize(img, verbose=False, sample_name=f"{sample_name}")

            data.imsave(f"{fpath_output}/fitted_imgs/{sample_name}.png")
            data.save_operations(f"{fpath_output}/operations/{sample_name}.png")
            data.save_wells(f"{fpath_output}/well_imgs/", sample_name, '.png')
        except:
            logging.error(f"Couldn't analyze sample {fpath_curr_img}", exc_info=True)
            bad_analysis_log.write(f"Couldn't analyze sample:\n    {fpath_curr_img}")
    def joblib_loop():
        Parallel(n_jobs=n_jobs)(delayed(analyze_phenotype)(i) for i in fpath_imgs)
    joblib_loop()
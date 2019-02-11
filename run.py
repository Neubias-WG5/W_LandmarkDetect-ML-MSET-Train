# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2016. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


__author__ = "Vandaele Rémy <remy.vandaele@ulg.ac.be>"
__contributors__ = ["Marée Raphaël <raphael.maree@ulg.ac.be>"]
__copyright__ = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"

from ldmtools import *
import sys
import scipy.ndimage as snd
from multiprocessing import Pool
from download import *
from cytomine.models import Annotation, Job, ImageInstanceCollection, AnnotationCollection, Property, AttachedFileCollection, AttachedFile
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
import numpy as np
import imageio
from subprocess import call
from neubiaswg5 import CLASS_LNDDET
from neubiaswg5.helpers import NeubiasJob, prepare_data, upload_data, upload_metrics, get_discipline

def	getcoordsim_neubias(gt_path, id_term, tr_im):
	xcs = []
	ycs = []
	xrs = []
	yrs = []
	for i in range(len(tr_im)):
		id = tr_im[i]
		gt_img = imageio.imread(os.path.join(gt_path, '%d.tif'%id))
		(y, x) = np.where(gt_img==id_term)
		(h, w) = gt_img.shape
		yc = y[0]
		xc = x[0]
		yr = yc/h
		xr = xc/w
		xcs.append(xc)
		ycs.append(yc)
		xrs.append(xr)
		yrs.append(yr)
	return np.array(xcs), np.array(ycs), np.array(xrs), np.array(yrs)

def main():
	with NeubiasJob.from_cli(sys.argv) as conn:
		problem_cls = get_discipline(conn, default=CLASS_LNDDET)
		is_2d = True
		conn.job.update(status=Job.RUNNING, progress=0, statusComment="Initialization of the training phase")
		in_images, gt_images, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, conn, is_2d=is_2d, **conn.flags)
		tmax = 1
		for f in os.listdir(gt_path):
			if f.endswith('.tif'):
				gt_img = imageio.imread(os.path.join(gt_path, f))
				tmax = np.max(gt_img)
				break

		term_list = range(1, tmax+1)
		depths = 1. / (2. ** np.arange(conn.parameters.model_depth))

		tr_im = [int(image.rstrip('.tif')) for image in os.listdir(in_path) if image.endswith('.tif')]

		DATA = None
		REP = None
		be = 0
		sfinal = ""
		for id_term in term_list:
			sfinal += "%d " % id_term
		sfinal = sfinal.rstrip(' ')
		for id_term in conn.monitor(term_list, start=10, end=90, period=0.05, prefix="Model building for terms..."):
			(xc, yc, xr, yr) = getcoordsim_neubias(gt_path, id_term, tr_im)
			nimages = np.max(xc.shape)
			mx = np.mean(xr)
			my = np.mean(yr)
			P = np.zeros((2, nimages))
			P[0, :] = xr
			P[1, :] = yr
			cm = np.cov(P)
			passe = False
			# additional parameters
			feature_parameters = None
			if conn.parameters.model_feature_type.lower() == 'gaussian':
				std_matrix = np.eye(2) * (conn.parameters.model_feature_gaussian_std ** 2)
				feature_parameters = np.round(np.random.multivariate_normal([0, 0], std_matrix, conn.parameters.model_feature_gaussian_n)).astype(int)
			elif conn.parameters.model_feature_type.lower() == 'haar':
				W = conn.parameters.model_wsize
				n = conn.parameters.model_feature_haar_n / (5 * conn.parameters.model_depth)
				h2 = generate_2_horizontal(W, n)
				v2 = generate_2_vertical(W, n)
				h3 = generate_3_horizontal(W, n)
				v3 = generate_3_vertical(W, n)
				sq = generate_square(W, n)
				feature_parameters = (h2, v2, h3, v3, sq)

			for times in range(conn.parameters.model_ntimes):
				if times == 0:
					rangrange = 0
				else:
					rangrange = conn.parameters.model_angle

				T = build_datasets_rot_mp(in_path, tr_im, xc, yc, conn.parameters.model_R, conn.parameters.model_RMAX, conn.parameters.model_P, conn.parameters.model_step, rangrange, conn.parameters.model_wsize, conn.parameters.model_feature_type, feature_parameters, depths, nimages, 'tif', conn.parameters.model_njobs)
				for i in range(len(T)):
					(data, rep, img) = T[i]
					(height, width) = data.shape
					if not passe:
						passe = True
						DATA = np.zeros((height * (len(T) + 100) * conn.parameters.model_ntimes, width))
						REP = np.zeros(height * (len(T) + 100) * conn.parameters.model_ntimes)
						b = 0
						be = height
					DATA[b:be, :] = data
					REP[b:be] = rep
					b = be
					be = be + height

			REP = REP[0:b]
			DATA = DATA[0:b, :]

			clf = ExtraTreesClassifier(n_jobs=conn.parameters.model_njobs, n_estimators=conn.parameters.model_ntrees)
			clf = clf.fit(DATA, REP)

			parameters_hash = {}
			parameters_hash['cytomine_id_terms'] = sfinal.replace(' ', ',')
			parameters_hash['model_R'] = conn.parameters.model_R
			parameters_hash['model_RMAX'] = conn.parameters.model_RMAX
			parameters_hash['model_P'] = conn.parameters.model_P
			parameters_hash['model_npred'] = conn.parameters.model_npred
			parameters_hash['model_ntrees'] = conn.parameters.model_ntrees
			parameters_hash['model_ntimes'] = conn.parameters.model_ntimes
			parameters_hash['model_angle'] = conn.parameters.model_angle
			parameters_hash['model_depth'] = conn.parameters.model_depth
			parameters_hash['model_step'] = conn.parameters.model_step
			parameters_hash['window_size'] = conn.parameters.model_wsize
			parameters_hash['feature_type'] = conn.parameters.model_feature_type
			parameters_hash['feature_haar_n'] = conn.parameters.model_feature_haar_n
			parameters_hash['feature_gaussian_n'] = conn.parameters.model_feature_gaussian_n
			parameters_hash['feature_gaussian_std'] = conn.parameters.model_feature_gaussian_std

			model_filename = joblib.dump(clf, os.path.join(out_path, '%d_model.joblib' % (id_term)), compress=3)[0]
			cov_filename = joblib.dump([mx, my, cm], os.path.join(out_path, '%d_cov.joblib' % (id_term)), compress=3)[0]
			parameter_filename = joblib.dump(parameters_hash, os.path.join(out_path, '%d_parameters.joblib' % id_term), compress=3)[0]
			AttachedFile(
				conn.job,
				domainIdent=conn.job.id,
				filename=model_filename,
				domainClassName="be.cytomine.processing.Job"
			).upload()
			AttachedFile(
				conn.job,
				domainIdent=conn.job.id,
				filename=cov_filename,
				domainClassName="be.cytomine.processing.Job"
			).upload()
			AttachedFile(
				conn.job,
				domainIndent=conn.job.id,
				filename=parameter_filename,
				domainClassName="be.cytomine.processing.Job"
			).upload()
			if conn.parameters.model_feature_type == 'haar' or conn.parameters.model_feature_type == 'gaussian':
				add_filename = joblib.dump(feature_parameters, out_path.rstrip('/')+'/'+'%d_fparameters.joblib' % (id_term))[0]
				AttachedFile(
					conn.job,
					domainIdent=conn.job.id,
					filename=add_filename,
					domainClassName="be.cytomine.processing.Job"
				).upload()

		Property(conn.job, key="id_terms", value=sfinal.rstrip(" ")).save()
		conn.job.update(progress=100, status=Job.TERMINATED, statusComment="Job terminated.")

if __name__ == "__main__":
	main()
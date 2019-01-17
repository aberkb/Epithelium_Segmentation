#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 16:11:28 2018

@author: aberkb
"""

from __future__ import print_function

from keras.callbacks import ModelCheckpoint
import numpy as np
from data import load_train_data
from preprocess import *
from model import get_unet
# Data definition
img_rows = 64 * 3
img_cols = 80 * 3

nb_total = 2000
nb_train = 1600
nb_labeled = 600
nb_unlabeled = nb_train - nb_labeled

apply_edt = True
nb_iterations = 10
nb_step_predictions = 20
nb_no_detections = 10
nb_random = 15
nb_most_uncertain = 10
most_uncertain_rate = 5
pseudo_epoch = 5
nb_pseudo_initial = 20
pseudo_rate = 20
initial_train = True
apply_augmentation = False
nb_initial_epochs = 10
nb_active_epochs = 2
batch_size = 128

X_train, y_train = load_train_data()
labeled_index = np.arange(0, nb_labeled)
unlabeled_index = np.arange(nb_labeled, len(X_train))

model = get_unet(dropout=True)
if initial_train:
    model_checkpoint = ModelCheckpoint(initial_weights_path, monitor='loss', save_best_only=True)

    if apply_augmentation:
        for initial_epoch in range(0, nb_initial_epochs):
            history = model.fit_generator(
                data_generator().flow(X_train[labeled_index], y_train[labeled_index], batch_size=32, shuffle=True),
                steps_per_epoch=len(labeled_index), nb_epoch=1, verbose=1, callbacks=[model_checkpoint])

            model.save(initial_weights_path)
            log(history, initial_epoch, log_file)
    else:
        history = model.fit(X_train[labeled_index], y_train[labeled_index], batch_size=32, nb_epoch=nb_initial_epochs,
                            verbose=1, shuffle=True, callbacks=[model_checkpoint])

        log(history, 0, log_file)
else:
    model.load_weights(initial_weights_path)

for iteration in range(1, nb_iterations + 1):
    if iteration == 1:
        weights = initial_weights_path

    else:
        weights = final_weights_path

    X_labeled_train, y_labeled_train, labeled_index, unlabeled_index = compute_train_sets(X_train, y_train,
                                                                                          labeled_index,
                                                                                          unlabeled_index, weights,
                                                                                          iteration)

    history = model.fit(X_labeled_train, y_labeled_train, batch_size=32, nb_epoch=nb_active_epochs, verbose=1,
                        shuffle=True, callbacks=[model_checkpoint])

    log(history, iteration, log_file)
    model.save(global_path + "models/active_model" + str(iteration) + ".h5")

log_file.close()

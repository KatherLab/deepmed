Training and Deploying a Simple Model
=====================================

TODO describe experiment file.


Experiment Imports
------------------

To do anything, we first have to import all the necessary functionality.  This
is easily done by writing::

    from deepest_histology.experiment_imports import *

at the top of our file.


Defining the Cohorts
--------------------

In the Deepest Histology pipeline, both training and deployment is performed on
*cohorts* of patients.

We will now a sets of cohorts one to train our data on::

    #TODO make this different cohorts
    train_cohorts = {
        Cohort(tile_path='E:/TCGA-BRCA-DX/BLOCKS_NORM',
               clini_path='G:/immunoproject/TCGA-IMMUNO_Clini_Slide/TCGA-BRCA-IMMUNO_CLINI.xlsx',
               slide_path='G:/immunoproject/TCGA-IMMUNO_Clini_Slide/TCGA-BRCA_SLIDE.csv'),
        Cohort(tile_path='E:/TCGA-BRCA-DX/BLOCKS_NORM',
               clini_path='G:/immunoproject/TCGA-IMMUNO_Clini_Slide/TCGA-BRCA-IMMUNO_CLINI.xlsx',
               slide_path='G:/immunoproject/TCGA-IMMUNO_Clini_Slide/TCGA-BRCA_SLIDE.csv')
    }

When using Windows-like paths with backslashes, this string ought to be prefixed
with an ``r`` to prevent the backslashes to be interpreted as character escapes:
``tile_path=r'C:\tile\path'``.

TODO describe clini / slide table.


Defining our Training Runs
--------------------------

Next, we have to define how we want to use these cohorts.  This is done using
so-called ``RunGetters``.  ``RunGetters`` allow us to define how we want to use
our data to train models.  We may for example want to perform cross-validation,
train only on certain subgroups, train on many targets or even do a combination
of the above.  For this example, we will settle for a simple, single-target
training.  Let's construct our simple run getter::

    simple_train_get = partial(
        get.simple_run,
        target_label='isMSIH',
        train_cohorts=train_cohorts,
        max_tile_num=100,
        na_values=['inconclusive'])

That is quite a lot to take in!  Let's break it down line by line.

*   ``get.simple_run`` describes how we want to use our data; in this case we
    want to train a simple, single-target model.  All the following lines
    describe how we want this training to be performed.
*   ``target_label='isMSIH'`` is the label we want to predict with our model.
    The clinical table is expected to have a column with that name.
*   ``train_cohorts=train_cohorts`` are the cohorts we want to use for training.
*   ``max_tile_num=100`` states how many of a patient's tiles we want to sample.
    Often times, increasing the number of tiles for a patient has only a minor
    effect on the actual training result.  Thus sampling from a patient's tiles
    can significantly speed up training without hugely influencing our results.
*   ``na_values=['inconclusive']`` allows us to define additional values which
    indicate a non-informational training sample.  Patients with this label will
    be excluded from training.


Training the Model
------------------

We can now finally train our model::

    if __name__ == '__main__':  # required on Windows
        do_experiment(
            project_dir='/path/to/training/project/dir',
            get=simple_get)

*   ``project_dir='/path/to/training/project/dir'`` defines where we want to
    save our training's results.  

And that's it!  Our model should now be merrily training!


Deploying the Model
-------------------

After our model has finished training, we may want to deploy it on another
dataset to ascertain its performance.  This is done quite similarly to the
training process.  We already `defined our test cohorts above`_, 
off by defining another run getter::

    # file: simple_deploy.py

    from deepest_histology.experiment_imports import *

    test_cohorts = {
        Cohort(tile_path='E:/TCGA-BRCA-DX/BLOCKS_NORM',
               clini_path='G:/immunoproject/TCGA-IMMUNO_Clini_Slide/TCGA-BRCA-IMMUNO_CLINI.xlsx',
               slide_path='G:/immunoproject/TCGA-IMMUNO_Clini_Slide/TCGA-BRCA_SLIDE.csv')
    }

    simple_deploy_get = partial(
        get.simple_run,
        target_label='isMSIH',
        test_cohorts=test_cohorts,
        max_tile_num=100,
        na_values=['inconclusive'])

Observe how the getter has the exact same structure as the one above, with the
only exception being that we specify ``test_cohorts`` instead of
``train_cohorts`` this time around.

We can now deploy our model like this::

    if __name__ == '__main__':
        do_experiment(
            project_dir='/path/to/deployment/project/dir',
            get=simple_deploy_get,
            model_path='/path/to/training/project/dir/export.pkl')

The ``model_path`` argument points to the saved trained model, which should
reside in the training project's directory.

.. _defined our test cohorts above: #defining-the-cohorts


Defining Evaluation Metrics
---------------------------

While our model has now been deployed on the testing cohort, we don't have any
results yet: this is because we haven't defined any metrics with which to
evaluate our testing data.  Let's start off with some simple metrics::

    evaluators = [auroc, count]

These metrics will calculate the `area under the receiver operating
characteristic curve`_ (AUROC) and the count of testing samples.  These metrics
are calculated on a *tile basis* though.  It is often advantagous to instead
calculate metrics on a per-patient basis instead.  This can be done with the
``Grouped`` adapter::

    evaluators += [Grouped(auroc, by='PATIENT'), Grouped(count, by='PATIENT')]

This will modify the auroc and count metrics in such a way that they are
calculated on a *per-patient* basis instead of a per-tile basis; instead of the
overall tile count per class we for example get the number of patients per
class.

If we now extend our deployment script to make use of these evaluators,
re-running the script should yield a file ``stats.csv`` which contains the
requested metrics::

    if __name__ == '__main__':
        do_experiment(
            project_dir='/path/to/deployment/project/dir',
            get=simple_deploy_get,
            model_path='/path/to/training/project/dir/export.pkl',
            evaluator_groups=[evaluators])

.. _area under the receiver operating characteristic curve: https://en.wikipedia.org/wiki/Receiver_operating_characteristic